---
name: amagi-inference-calc
description: Size GPU hardware for running a local LLM with llama.cpp. Works out VRAM (weights + per-user KV cache + runtime), decode tok/s per person when several people send requests at once, prompt-reading speed and time to first token, and used-GPU cost per tok/s. Ranks the cheapest card setups for small-office presets. Use when the user asks which GPU to buy for local AI, whether a model fits on a card, how fast a model will run for N people, whether a used card is worth its price, or how many cards a model needs.
---

# amagi-inference-calc

A CLI that estimates memory, speed and cost for a local LLM on a given GPU setup. Memory comes from the
model's real layout (layers, KV heads, sliding-window layers, MoE experts). Speed comes from a memory
bandwidth model fitted on real llama.cpp CUDA runs on one RTX 3090; other cards are scaled by bandwidth.
No network calls, no dependencies.

## When to use this skill

- "Which GPU do I need for a 10-person office?" Run `office`.
- "Will Qwen3 32B fit on my 4090, and how fast with 3 people?" Run `estimate`.
- "Is a used P40 worth it over a 3060?" Run `office` with `--gpu` for both, or `estimate` on each.
- The user has a `.gguf` file or a Hugging Face `config.json` and wants its KV cost per token. Run `gguf`
  or pass the file to `estimate`.
- The user has measured their own card (amagi-tokbench report or memtest_vulkan log) and wants
  estimates from those numbers instead of the spec sheet.

## How to use it

Install once: `pipx install amagi-inference-calc` (or `pip install -e .` from this repo in development).

```bash
amagi-inference-calc estimate qwen3-14b --gpu rtx-3090 --users 2 --context 8192 --prompt 2000
amagi-inference-calc estimate model.gguf --vram 16 --bw 448      # a card not in the list
amagi-inference-calc estimate llama-3.3-70b --gpu rtx-3090 --gpus 3
amagi-inference-calc office team-10                              # cheapest setups for a preset
amagi-inference-calc office doc-review --model qwen3-32b --all-vendors
amagi-inference-calc gpus                                        # bandwidth and used prices
amagi-inference-calc models                                      # built-in model ids
amagi-inference-calc calibrate --role backtest                   # how far off the model was
amagi-inference-calc calibrate --tokbench report.json            # fit the user's own card
amagi-inference-calc bandwidth memtest.log                       # memtest_vulkan GB/s
```

Add `--json` to any command for machine-readable output.

`--users` means people sending a request at the same moment, not staff count. The presets assume about a
quarter to a third of staff at once: `solo` 1, `team-5` 2, `team-10` 3, `team-25` 6, `doc-review` 2 at
32K context, `coding` 2 at 32K. Ask the user how many people they expect at once if the preset doesn't
fit.

## Reading the results

`estimate --json` returns one object. The fields that matter:

- `vram_needed_gib` / `vram_total_gib` / `fits`, plus `weights_gib` and `kv_gib`.
- `tok_s_single` (one person), `tok_s_per_user` (each of `users` at once), `tok_s_total`.
- `prefill_tok_s`, `ttft_s` (first token for one request), `ttft_all_s` (if all send at once). These
  are `null` on cards with no measured FLOPs figure; only the RTX 3090 has one. Say "not measured"
  rather than guessing. The user can pass `--tflops`.
- `bandwidth_source` says where the speed came from: calibrated, memtest, or `spec x 0.81`.
- `price_usd` and `usd_per_tok_s` use the bundled used prices (lowest asking price, dated in the
  output). Prices go stale; offer `--price GPU=USD` if the user has a real quote.
- `warnings` carries card caveats (old Pascal cards, non-NVIDIA cards on a CUDA calibration,
  multi-card layer split, context past the model's trained length). Always pass these on.

`office --json` returns `preset`, `min_tok_s`, `price_date` and `options`, a map from model id to a list
of setups in the same shape as `estimate`, sorted by price.

## How much to trust it

From `calibrate`: single-user speed on the 3090 came within about 8% on held-out models (leaning
optimistic), multi-user MoE within about 10%. Dense multi-user batching is not measured. Prompt reading
for MoE models is rough (-28% to +64%). Other cards are extrapolated by bandwidth, so give their numbers as
estimates, and say so. Multi-card setups assume llama.cpp layer split: memory adds up, speed does not.
The full method is in `docs/CALCULATIONS.md`.

When the user starts llama.cpp for N users with C tokens each, remind them that `-c` is shared across
`-np` slots: use `-c N*C -np N`.
