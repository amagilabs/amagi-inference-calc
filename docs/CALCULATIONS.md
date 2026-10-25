# How the numbers are worked out

This page covers every number the calculator shows: what goes in, the
formula, where the constants come from, and how far off the model was on
runs it never saw. The code is in `src/amagi_inference_calc/` (Python) and
`web/src/speed.ts` (the same model in TypeScript for the web page; a test
checks the two agree on more than 2,000 cases).

Short version: memory comes from the model's architecture and the file
format. Speed comes from memory bandwidth, because generating a token means
reading the weights once. The constants were fitted on one RTX 3090. Every
other card is scaled from it, so treat its numbers as estimates.

## 1. Memory

```
needed = weights + users x KV cache per user + 0.5 GiB runtime per card
```

**Weights.** `params x bits per weight / 8`. When the model entry was read
from a real GGUF file, the exact file size is used instead. Bits per weight
come from `quant.py`:

| Format | Bits/weight | How we know |
|---|---|---|
| F16 / BF16 | 16 | exact |
| Q8_0 | 8.5 | exact (34-byte block per 32 weights) |
| Q6_K | 6.5625 | exact (210-byte block per 256 weights) |
| Q5_K_M | 5.75 | estimate |
| Q4_K_M | 4.88 | measured on two files (Qwen2.5-7B 4.91, Qwen2.5-32B 4.85) |
| Q4_0 | 4.5 | exact (18-byte block per 32 weights) |
| MXFP4, IQ4_XS | 4.25 | exact |
| Q3_K_M | 3.91 | estimate |

The `_M` and `_S` mixes keep some tensors at a higher precision, which is why
they come out above their block size. The ones marked estimate have not been
checked against a real file.

**KV cache.** For one user with a context of `C` tokens:

```
per token per layer = kv_heads x (head_dim_k + head_dim_v) x bytes per element
KV = full-attention layers x per_token_layer x C
   + sliding-window layers x per_token_layer x min(C, window)
```

F16 is 2 bytes per element, Q8_0 is 34/32 and Q4_0 is 18/32. Sliding-window
layers (Gemma 3, gpt-oss) only keep the last `window` tokens, which is why
those models need much less KV than the head count suggests. The calculator
reserves KV for each user's full context. In llama.cpp, `-c` is shared
across the `-np` slots, so to give N users C tokens each, start the server
with `-c` set to N x C and `-np N`.

**Runtime.** 0.5 GiB per card for the CUDA context and compute buffers. On
the 3090 we measured 0.11 to 0.46 GiB.

**Unified memory** (Apple M-series, AMD Ryzen AI Max+ 395): 75% of system
memory is treated as usable by the GPU. That is an assumption; the real limit
depends on the OS and its settings.

**More than one card.** Memory adds up. Speed does not, because the
calculator assumes llama.cpp's default layer split, where the cards take
turns. So the cheapest-setup list never adds a card once the model fits.

The **Memory only** tab is Alex Ziskind's original calculator, unchanged:
`params x GB per billion for the format`, plus a context term that scales
with `context / 2048`, all times 1.1. The GPU count is
`ceil(VRAM needed x 1.2 / card VRAM)`.
`tests/test_upstream_parity.py` runs the original TypeScript and checks the
Python port matches it on 1,152 cases.

## 2. Speed per person (decode)

Generating one token means reading every active weight once, plus each
user's KV cache. With `N` people sending at the same moment, one decode step
produces one token for each of them:

```
step = (weight bytes read + N x KV bytes) / effective bandwidth
       + overhead
       + moe_layer_ms x MoE layers
       + (N - 1) x cost per extra stream

speed per person = 1 / step          (tok/s)
all N together   = N / step
```

- **Weight bytes read.** For a dense model, all the weights. For a
  mixture-of-experts model, the shared weights plus the experts that get
  used. With `N` streams in one step, the share of experts touched is
  `1 - (1 - used/total)^N`, so a batch reads more of the model than one
  stream does.
- **KV bytes** are counted at the prompt length used for the speed numbers
  (the "typical request" on the web page), not at the full context.
- **Cost per extra stream.** For MoE models this was fitted on Qwen3.6 35B-A3B
  at 4 and 8 streams. For dense models it is the extra matmul work at the
  measured prefill rate. **That dense number has not been checked against a
  real multi-user run.**

## 3. Wait before the answer starts (prefill)

Reading the prompt is limited by compute, not bandwidth:

```
prefill time = 2 x active params x prompt tokens / F
             + attention FLOPs / F_dense
```

`F` is the effective dense TFLOPS for dense models and a separate, lower
figure for MoE expert matmuls. Attention FLOPs grow with the square of the
prompt length on full-attention layers, and linearly once a sliding window
caps them.

"Up to X if all N send at once" is `prefill time x N`: the worst case, where
every request lands in the same second and waits its turn.

Prefill needs a FLOPs number for the card, and only the RTX 3090 has one
so far. Every other card shows **not measured**. The command line accepts
`--tflops` if you know your card's effective rate.

## 4. Bandwidth for cards we have not measured

In order of preference:

1. **Your own run.** `amagi-inference-calc calibrate --tokbench report.json`
   fits decode bandwidth from a real benchmark on your card.
2. **The calibration card** (RTX 3090): 758 GB/s, fitted.
3. **A memory test.** If a card has a `memtest_vulkan` figure, it is scaled by
   the 3090's ratio of decode bandwidth to memtest (758 / 737.8 = 1.03).
4. **The spec sheet.** Vendor bandwidth x 0.81, the 3090's ratio of decode
   bandwidth to its 936 GB/s spec.

AMD, Intel and Apple cards go through route 4 on a CUDA calibration, so
their real numbers often come in lower. The tool says so next to each result.

## 5. Price and cost per tok/s

```
hardware cost = used price per card x cards
cost per tok/s = hardware cost / (speed per person x N)
```

Prices are the lowest used listing on gpudojo.com (Amazon, eBay and Newegg
offers), fetched 2026-10-25. They are asking prices, not sold prices, and
they move every week. Type your own price into the page, or pass `--price`
or `--prices FILE` on the command line.

## 6. Office presets

`users` means people sending a request at the same moment, not head count.
The presets assume a quarter to a third of staff hit enter at the same time.

| Preset | Staff | At once | Context | Typical request |
|---|---|---|---|---|
| Solo / owner | 1 | 1 | 8K | 1,500 |
| 5-person office | 5 | 2 | 8K | 2,000 |
| 10-person office | 10 | 3 | 16K | 3,000 |
| 25-person office | 25 | 6 | 16K | 3,000 |
| Document review (law, accounting) | 5 | 2 | 32K | 20,000 |
| Small dev team | 4 | 2 | 32K | 12,000 |

**Cheapest setups.** For each model, try every priced card at 1 to 4 copies.
Keep the first count that fits, but only if it also reaches the minimum
speed per person (15 tok/s by default; people read about 5 tokens a second).
Then sort by price. NVIDIA only unless you include AMD and Intel.

## 7. Where the constants come from

All fitted with `speed.fit()` from `data/measurements.json`. Every run in
that file used one GeForce RTX 3090 24GB at a 300 W power limit, all layers
on the GPU, f16 KV cache, llama.cpp CUDA.

| Constant | Value | Fitted from |
|---|---|---|
| Effective decode bandwidth | 758.0 GB/s | dense single-stream decode |
| Overhead per step | 1.29 ms | dense single-stream decode |
| Extra per MoE layer | 0.050 ms | MoE single-stream decode |
| Extra per stream per MoE layer | 0.058 ms | MoE decode at 4 and 8 streams |
| Extra per stream per 1B dense params | 0.030 ms | derived from dense prefill FLOPs (not measured) |
| Dense prefill | 67.8 TFLOPS | dense prefill |
| MoE expert prefill | 29.4 TFLOPS | MoE prefill |
| Memtest reference | 737.8 GB/s | memtest_vulkan 0.6.0, median of 30 samples on each of two cards (739.3 and 736.2) |

Run `amagi-inference-calc calibrate` to print the table below yourself, or
`calibrate --refit` to fit again from the same file.

## 8. Back-test: how far off is it?

"fit" rows set the constants. "backtest" rows were held out and only used
to measure error. tok/s is per stream.

| Kind | Model | Setting | Role | Measured | Predicted | Error |
|---|---|---|---|---|---|---|
| decode | qwen2.5-7b | ctx 198, 1 stream | fit | 141.9 | 141.4 | -0.3% |
| decode | qwen2.5-7b | ctx 2534, 1 stream | fit | 137.3 | 137.9 | +0.5% |
| decode | qwen2.5-32b | ctx 197, 1 stream | fit | 37.0 | 37.1 | +0.1% |
| decode | qwen2.5-32b | ctx 2535, 1 stream | fit | 36.1 | 36.0 | -0.3% |
| decode | qwen3.6-35b-a3b | ctx 300, 1 stream | fit | 148.7 | 147.6 | -0.8% |
| decode | qwen3.6-35b-a3b | ctx 41782, 1 stream | fit | 125.8 | 126.6 | +0.7% |
| decode | qwen3.8-27b | ctx 300, 1 stream | backtest | 42.8 | 46.3 | +8.1% |
| decode | qwen3.8-27b | ctx 41782, 1 stream | backtest | 37.6 | 39.7 | +5.5% |
| decode | gpt-oss-20b | ctx 361, 1 stream | backtest | 186.3 | 198.3 | +6.4% |
| decode | gpt-oss-20b | ctx 37462, 1 stream | backtest | 153.5 | 160.1 | +4.3% |
| decode | gpt-oss-20b | ctx 56435, 1 stream | backtest | 139.6 | 145.8 | +4.4% |
| decode | qwen3.6-35b-a3b | ctx 2600, 4 streams | fit | 66.4 | 61.5 | -7.3% |
| decode | qwen3.6-35b-a3b | ctx 2600, 8 streams | fit | 31.9 | 35.1 | +9.9% |
| decode | gpt-oss-20b | ctx 2400, 4 streams | backtest | 79.1 | 74.8 | -5.5% |
| decode | gpt-oss-20b | ctx 2400, 8 streams | backtest | 41.9 | 44.5 | +6.2% |
| prefill | qwen2.5-7b | prompt 2470 | fit | 4,497 | 4,631 | +3.0% |
| prefill | qwen2.5-32b | prompt 2471 | fit | 1,064 | 1,034 | -2.8% |
| prefill | qwen3.8-27b | prompt 41782 | backtest | 1,201 | 1,140 | -5.1% |
| prefill | qwen3.6-35b-a3b | prompt 41782 | fit | 3,049 | 3,981 | +30.6% |
| prefill | qwen3.6-35b-a3b | prompt 2388 | backtest | 3,004 | 4,913 | +63.5% |
| prefill | gpt-oss-20b | prompt 37334 | fit | 4,307 | 3,331 | -22.7% |
| prefill | gpt-oss-20b | prompt 9307 | backtest | 5,321 | 3,853 | -27.6% |

(`calibrate` prints all 28 rows; a few near-duplicate gpt-oss rows are left
out here.)

What that means:

- **Single-user speed on the 3090:** within about 8% on models it never saw,
  and it leans optimistic (+4% to +8%).
- **Several users, MoE:** within about 10%, fitted and held out.
- **Several users, dense:** not tested. The batching cost comes from FLOPs
  arithmetic only.
- **Prompt reading, dense:** within about 5%.
- **Prompt reading, MoE:** rough. Errors run from -28% to +64%, so read the
  first-token wait for MoE models as a ballpark.
- **Other cards:** these scale from the 3090 by bandwidth. Nobody has
  measured them here, so expect larger errors, especially on older cards
  (Pascal, such as the Tesla P40) and on non-NVIDIA hardware.

## 9. Known limits

- One calibration card, one runtime (llama.cpp CUDA). vLLM, ExLlama, MLX and
  ROCm will land somewhere else.
- More than 8 people at once is past the measured range (1 to 8 streams).
- Layer split only. Tensor parallel (vLLM `-tp`) can make 2 cards faster
  than 1; this tool does not model that.
- Models with no GGUF source have their architecture numbers computed from
  the public `config.json`, so their file sizes are estimates.
- Contexts longer than a model was trained on get a warning, not a block.

Found a measurement that disagrees? Open an issue with the command, the
card, the llama.cpp build and the numbers. Real runs make the fit better.
