"""amagi-inference-calc command line."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from . import __version__, price as P, speed as S, upstream as U
from .data import load_json
from .models import builtin_models, find_model

OFFICE_MODELS = ("qwen3-8b", "qwen3-14b", "gpt-oss-20b", "gemma-3-27b", "qwen3-30b-a3b", "qwen3-32b", "llama-3.3-70b")


def _num(x: Optional[float], fmt: str = "{:,.1f}") -> str:
    return "-" if x is None else fmt.format(x)


def _usd(x: Optional[float]) -> str:
    return "-" if x is None else f"${x:,.0f}"


def _emit(obj, as_json: bool) -> bool:
    if as_json:
        print(json.dumps(obj, indent=2))
    return as_json


def _table(rows: List[Sequence[str]], header: Sequence[str], left: int = 1) -> str:
    """Plain-text table; the first `left` columns are left-aligned, the rest right."""
    cols = list(zip(*([header] + [list(r) for r in rows]))) if rows else [[h] for h in header]
    widths = [max(len(str(c)) for c in col) for col in cols]

    def line(r):
        return "  ".join(str(c).ljust(w) if i < left else str(c).rjust(w)
                         for i, (c, w) in enumerate(zip(r, widths))).rstrip()
    return "\n".join([line(header)] + [line(r) for r in rows])


def _guess_model(name: str):
    """Match a file name like Qwen2.5-7B-Instruct-Q4_K_M.gguf to a built-in model."""
    try:
        return find_model(name)
    except KeyError:
        pass
    parts = name.lower().replace(".gguf", "").replace("_", "-").split("-")
    for n in range(len(parts) - 1, 0, -1):
        try:
            return find_model("-".join(parts[:n]))
        except KeyError:
            continue
    raise KeyError(f"cannot tell which model {name!r} is; pass --model")


# -- gpu / price plumbing -----------------------------------------------------

def _gpu_from_args(a) -> S.GPU:
    if a.gpu:
        g = S.find_gpu(a.gpu)
    elif a.vram and a.bw:
        g = S.custom_gpu(a.vram, a.bw)
    else:
        raise SystemExit("error: give --gpu NAME, or --vram GB and --bw GB/s for a card that is not in the list")
    if getattr(a, "vram", None) and a.gpu:
        g.vram_gb = a.vram
    if getattr(a, "bw", None) and a.gpu:
        g.bw_measured_gbs = a.bw  # a memtest number for this card
    if getattr(a, "decode_bw", None):
        g.bw_decode_gbs = a.decode_bw
    return g


def _prices(a) -> dict:
    raw = P.parse_overrides(getattr(a, "price_pair", None) or (), getattr(a, "prices", None))
    # accept the same short names as --gpu ("3090" -> rtx-3090); an unknown name is an error, not a no-op
    return {S.find_gpu(k).id: v for k, v in raw.items()}


def _calibration(a) -> S.Calibration:
    path = getattr(a, "calibration", None)
    if not path:
        return S.load_calibration()
    with open(path, encoding="utf-8") as fh:
        c = json.load(fh)
    fields = S.Calibration.__dataclass_fields__
    return S.Calibration(**{k: v for k, v in c.items() if k in fields})


def _add_gpu_args(p: argparse.ArgumentParser, gpu_required: bool = False) -> None:
    p.add_argument("--gpu", "-g", required=gpu_required, help="GPU id or name (see `gpus`)")
    p.add_argument("--gpus", "-n", type=int, default=1, help="number of identical cards (layer split; default 1)")
    p.add_argument("--vram", type=float, help="VRAM in GB, for a card not in the list")
    p.add_argument("--bw", type=float, help="memory bandwidth in GB/s (spec, or a memtest_vulkan number)")
    p.add_argument("--decode-bw", type=float, help="effective decode bandwidth from `calibrate --tokbench`")
    p.add_argument("--tflops", type=float, help="effective prefill TFLOPS, to get prompt speed on uncalibrated GPUs")


def _add_price_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--price", dest="price_pair", action="append", metavar="GPU=USD",
                   help="override a used price, e.g. --price rtx-3090=1100 (repeatable)")
    p.add_argument("--prices", metavar="CSV", help="CSV of gpu,usd rows to override prices")


def _add_load_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--users", "-u", type=int, default=1, help="people sending requests at the same moment (default 1)")
    p.add_argument("--context", "-c", type=int, default=8192, help="context window per user, tokens (default 8192)")
    p.add_argument("--prompt", type=int, help="prompt length for speed numbers (default: half the context)")
    p.add_argument("--quant", "-q", help="weight quant, e.g. Q4_K_M, Q8_0, MXFP4 (default: model native or Q4_K_M)")
    p.add_argument("--kv", default="F16", help="KV cache type: F16, Q8_0, Q4_0 (default F16)")


# -- commands -------------------------------------------------------------------

def _print_estimate(e: S.Estimate, model_name: str, gpu: S.GPU, own_price: bool = False) -> None:
    cards = f"{e.gpu_count}x " if e.gpu_count > 1 else ""
    print(f"{model_name} ({e.quant}) on {cards}{gpu.name}")
    print(f"  load       {e.users} user(s) at once, {e.context:,}-token context each")
    verdict = "fits" if e.fits else f"does NOT fit (needs {e.cards_needed} card(s))"
    print(f"  memory     {e.vram_needed_gib:.1f} GiB needed of {e.vram_total_gib:.1f} GiB -> {verdict}")
    overhead = e.vram_needed_gib - e.weights_gib - e.kv_gib
    print(f"             weights {e.weights_gib:.1f} + KV cache {e.kv_gib:.1f} + runtime {overhead:.1f}")
    if e.users > 1:
        print(f"  speed      {e.tok_s_single:.1f} tok/s for one user; {e.tok_s_per_user:.1f} tok/s each with "
              f"{e.users} at once ({e.tok_s_total:.0f} total)")
    else:
        print(f"  speed      {e.tok_s_single:.1f} tok/s writing the answer (at a {e.prompt:,}-token conversation)")
    if e.prefill_tok_s:
        line = f"  prompt     {e.prefill_tok_s:,.0f} tok/s reading; first token after {e.ttft_s:.1f} s ({e.prompt:,}-token prompt)"
        if e.users > 1:
            line += f", {e.ttft_all_s:.1f} s if all {e.users} send at once"
        print(line)
    else:
        print("  prompt     not estimated for this GPU (pass --tflops to add it)")
    print(f"  bandwidth  {e.bandwidth_gbs:.0f} GB/s effective ({e.bandwidth_source})")
    if e.price_usd is not None:
        src = f"used, {P.price_date()}" if gpu.price_source and not own_price else "your price"
        per = f" -> ${e.usd_per_tok_s:,.2f} per tok/s" if e.usd_per_tok_s else ""
        print(f"  price      {_usd(e.price_usd)} ({src}){per}")
    for w in e.warnings:
        print(f"  note       {w}")


def cmd_estimate(a) -> int:
    model = find_model(a.model)
    gpu = _gpu_from_args(a)
    pr = _prices(a)
    price = a.usd if a.usd is not None else pr.get(gpu.id)
    e = S.estimate(model, gpu, quant=a.quant, users=a.users, context=a.context, prompt=a.prompt,
                   gpu_count=a.gpus, kv_type=a.kv, tflops=a.tflops, price=price, cal=_calibration(a))
    if _emit(e.to_dict(), a.json):
        return 0
    _print_estimate(e, model.name, gpu, own_price=a.usd is not None or gpu.id in pr)
    return 0


def _office_rows(preset: dict, models: Sequence[str], a) -> List[dict]:
    pr = _prices(a)
    cal = _calibration(a)
    gpus = [g for g in S.builtin_gpus() if a.include_unpriced or g.price_used_usd is not None or g.id in pr]
    if not a.all_vendors:
        gpus = [g for g in gpus if g.vendor == "nvidia"]
    if a.gpu:
        gpus = [S.find_gpu(x) for x in a.gpu]
    rows = []
    for mid in models:
        m = find_model(mid)
        for g in gpus:
            for n in range(1, (1 if g.kind == "unified" else a.max_cards) + 1):
                e = S.estimate(m, g, quant=a.quant, users=preset["users"], context=preset["context"],
                               prompt=preset["prompt"], gpu_count=n, kv_type=a.kv, price=pr.get(g.id), cal=cal)
                if not e.fits:
                    continue
                if e.tok_s_per_user >= a.min_tok_s:
                    d = e.to_dict()
                    d["model_name"], d["gpu_name"] = m.name, g.name
                    rows.append(d)
                break  # more cards never makes a layer split faster
    return rows


def cmd_office(a) -> int:
    presets = {p["id"]: p for p in load_json("office_presets.json")["presets"]}
    if a.preset not in presets:
        raise SystemExit(f"error: unknown preset {a.preset!r}; choose from {', '.join(presets)}")
    preset = dict(presets[a.preset])
    for k in ("users", "context", "prompt"):
        if getattr(a, k) is not None:
            preset[k] = getattr(a, k)
    rows = _office_rows(preset, a.model or OFFICE_MODELS, a)
    key = (lambda r: (r["price_usd"] is None, r["price_usd"] or 0, -r["tok_s_per_user"]))
    rows.sort(key=key)
    best = {}
    for r in rows:
        best.setdefault(r["model"], []).append(r)
    if _emit({"preset": preset, "min_tok_s": a.min_tok_s, "price_date": P.price_date(),
              "options": {k: v[: a.top] for k, v in best.items()}}, a.json):
        return 0
    print(f"{preset['name']}: {preset['users']} user(s) at once, {preset['context']:,}-token context, "
          f"~{preset['prompt']:,}-token prompts")
    print(f"Cheapest setups giving each user at least {a.min_tok_s:g} tok/s (used prices, {P.price_date()}):\n")
    out = []
    for mid in (a.model or OFFICE_MODELS):
        m = find_model(mid)
        opts = best.get(m.id, [])
        if not opts:
            out.append([m.id, "nothing in the list fits at this speed", "", "", "", "", ""])
            continue
        for r in opts[: a.top]:
            setup = (f"{r['gpu_count']}x " if r["gpu_count"] > 1 else "") + r["gpu"]
            if any(r["warnings"]) and not all(w.startswith(("multi-GPU", "more than 8")) for w in r["warnings"]):
                setup += " *"
            out.append([m.id, setup, f"{r['vram_needed_gib']:.1f}/{r['vram_total_gib']:.0f} GiB",
                        _num(r["tok_s_per_user"]), _num(r["tok_s_total"], "{:,.0f}"),
                        _usd(r["price_usd"]), _num(r["usd_per_tok_s"], "${:,.2f}")])
    print(_table(out, ["model", "setup", "memory", "tok/s each", "total", "price", "$ per tok/s"], left=2))
    print("\nSpeeds are estimates from a model calibrated on one RTX 3090 (see `calibrate`).")
    if any(r[1].endswith(" *") for r in out):
        print("* this card has a caveat: run `estimate` on it to read the note.")
    if not a.all_vendors and not a.gpu:
        print("NVIDIA only; add --all-vendors to include AMD, Intel and Apple.")
    return 0


def cmd_gpus(a) -> int:
    pr = _prices(a)
    cal = S.load_calibration()
    gs = S.builtin_gpus()
    rows = []
    for g in gs:
        bw, src = S._bandwidth(g, cal)
        p = pr.get(g.id, g.price_used_usd)
        rows.append({"id": g.id, "name": g.name, "vram_gb": g.vram_gb, "bw_spec_gbs": g.bw_spec_gbs,
                     "bw_decode_gbs": round(bw / 1e9, 1), "bw_source": src, "price_used_usd": p,
                     "usd_per_gb": round(p / g.vram_gb, 1) if p else None})
    if _emit({"price_date": P.price_date(), "gpus": rows}, a.json):
        return 0
    print(_table([[r["id"], f"{r['vram_gb']:g}", f"{r['bw_spec_gbs']:g}", f"{r['bw_decode_gbs']:.0f}", _usd(r["price_used_usd"]),
                   _num(r["usd_per_gb"], "${:,.0f}")] for r in rows],
                 ["gpu", "VRAM GB", "spec GB/s", "decode GB/s", "used price", "$ per GB"]))
    print(f"\nPrices: lowest used listing on gpudojo.com, {P.price_date()} (asking, not sold). Unified-memory VRAM is 75% of RAM.")
    return 0


def cmd_models(a) -> int:
    ms = builtin_models()
    if _emit([dict(m.to_dict(), kv_mib_per_1k_tokens=round(m.kv_bytes(1000) / 2**20, 1)) for m in ms], a.json):
        return 0
    rows = [[m.id, f"{m.params / 1e9:.1f}", f"{m.active_params / 1e9:.1f}", str(m.layers),
             f"{m.kv_bytes(1000) / 2**20:.0f}", f"{m.context_max:,}",
             "GGUF header" if m.source.startswith("GGUF") else "HF config.json"] for m in ms]
    print(_table(rows, ["model", "params B", "active B", "layers", "KV MiB/1K tok", "max ctx", "source"]))
    print("\nAny GGUF file or Hugging Face config.json path also works wherever a model is asked for.")
    return 0


def cmd_gguf(a) -> int:
    from .gguf import read_gguf
    from .models import from_gguf
    info = read_gguf(a.file)
    spec = from_gguf(info)
    d = info.summary()
    d["kv_mib_per_1k_tokens_f16"] = round(spec.kv_bytes(1000) / 2**20, 2)
    d["read_gb_per_token"] = round(spec.read_bytes / 1e9, 3)
    if _emit(d, a.json):
        return 0
    for k, v in d.items():
        print(f"{k:28} {v}")
    return 0


def cmd_bandwidth(a) -> int:
    from .memtest import read_memtest
    r = read_memtest(a.log)
    if _emit(r.to_dict(), a.json):
        return 0
    print(f"{r.device} ({r.vram_gb} GB): {r.median_gbs:g} GB/s median over {r.passes - 1} passes "
          f"(range {r.min_gbs:g}-{r.max_gbs:g}){'  ERRORS REPORTED' if r.errors else ''}")
    print(f"use it with: amagi-inference-calc estimate MODEL --gpu YOUR-GPU --bw {r.median_gbs:g}")
    return 1 if r.errors else 0


def cmd_calibrate(a) -> int:
    if a.tokbench:
        with open(a.tokbench, encoding="utf-8") as fh:
            rep = json.load(fh)
        model = find_model(a.model) if a.model else _guess_model(rep.get("backend", {}).get("model_file", ""))
        quant = a.quant or rep.get("backend", {}).get("quant") or model.native_quant or "Q4_K_M"
        runs = S.tokbench_runs(rep)
        bw = S.decode_bw_from_runs(model, quant, runs)
        out = {"model": model.id, "quant": quant, "runs": runs, "decode_bw_gbs": round(bw, 1)}
        if _emit(out, a.json):
            return 0
        print(f"{len(runs)} single-user run(s) of {model.id} ({quant}): effective decode bandwidth {bw:.0f} GB/s")
        print(f"use it with: amagi-inference-calc estimate MODEL --gpu YOUR-GPU --decode-bw {bw:.0f}")
        return 0
    if a.refit:
        cal = S.fit()
        from dataclasses import asdict
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(cal).items()}
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=1)
        if _emit(d, a.json):
            return 0
        for k, v in d.items():
            if k != "notes":
                print(f"{k:24} {v}")
        return 0
    rows = S.backtest(role=a.role)
    if _emit(rows, a.json):
        return 0
    cal = S.load_calibration()
    print(f"Calibration: {cal.gpu}, decode {cal.bw_eff_gbs:.0f} GB/s + {cal.overhead_ms:.2f} ms/step, "
          f"prefill {cal.tflops_dense:.0f} TFLOPS dense / {cal.tflops_moe:.0f} MoE\n")
    print(_table([[r["kind"], r["model"], r["where"], r["role"], _num(r["measured"]), _num(r["predicted"]),
                   f"{r['error_pct']:+.1f}%"] for r in rows],
                 ["kind", "model", "where", "role", "measured", "predicted", "error"], left=4))
    return 0


def cmd_vram(a) -> int:
    rec = U.hardware_recommendation(a.params, a.quant, a.context, not a.no_kv_cache, a.kv_quant,
                                    a.memory_mode, a.system_memory, a.gpu_vram, a.mode)
    rec["onDiskSizeGB"] = round(U.on_disk_size(a.params, a.quant), 2)
    if _emit(rec, a.json):
        return 0
    for k, v in rec.items():
        print(f"{k:16} {v}")
    print("\n(upstream llm-inference-calculator rule of thumb; `estimate` uses the model's real layout)")
    return 0


# -- parser ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="amagi-inference-calc",
        description="How much GPU does a local LLM need, and how fast will it run? "
                    "VRAM, tokens/second and used-GPU cost per token/second, from real measurements.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="memory, speed and cost for one model on one setup")
    e.add_argument("model", help="model id (see `models`), GGUF file, or config.json")
    _add_gpu_args(e)
    _add_load_args(e)
    _add_price_args(e)
    e.add_argument("--usd", type=float, help="what you paid for one card (overrides the price list)")
    e.add_argument("--calibration", metavar="JSON", help="calibration file from `calibrate --refit --out`")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_estimate)

    o = sub.add_parser("office", help="cheapest GPU setups for a small-office preset")
    o.add_argument("preset", nargs="?", default="team-5",
                   help="solo, team-5, team-10, team-25, doc-review, coding (default team-5)")
    o.add_argument("--model", "-m", action="append", help="model to consider (repeatable; default: a common set)")
    o.add_argument("--gpu", "-g", action="append", help="only consider these GPUs (repeatable)")
    o.add_argument("--users", "-u", type=int, help="override the preset's users at once")
    o.add_argument("--context", "-c", type=int, help="override the preset's context")
    o.add_argument("--prompt", type=int, help="override the preset's prompt size")
    o.add_argument("--quant", "-q")
    o.add_argument("--kv", default="F16")
    o.add_argument("--min-tok-s", type=float, default=15.0, help="minimum tok/s per user (default 15)")
    o.add_argument("--max-cards", type=int, default=4, help="most identical cards to try (default 4)")
    o.add_argument("--top", type=int, default=3, help="setups to show per model (default 3)")
    o.add_argument("--include-unpriced", action="store_true", help="also list GPUs with no used price (Macs etc.)")
    o.add_argument("--all-vendors", action="store_true",
                   help="include AMD, Intel and Apple (speed less certain: the model is calibrated on NVIDIA CUDA)")
    o.add_argument("--calibration", metavar="JSON")
    _add_price_args(o)
    o.add_argument("--json", action="store_true")
    o.set_defaults(func=cmd_office)

    g = sub.add_parser("gpus", help="list GPUs with bandwidth and used prices")
    _add_price_args(g)
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=cmd_gpus)

    m = sub.add_parser("models", help="list built-in models")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_models)

    f = sub.add_parser("gguf", help="read a GGUF header: params, quant, KV size per token")
    f.add_argument("file")
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_gguf)

    b = sub.add_parser("bandwidth", help="measured GB/s from a memtest_vulkan log")
    b.add_argument("log")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_bandwidth)

    c = sub.add_parser("calibrate", help="show the back-test, refit, or fit your own tokbench run")
    c.add_argument("--tokbench", metavar="JSON", help="amagi-tokbench report from your own GPU")
    c.add_argument("--model", help="model of the tokbench run (default: from the report)")
    c.add_argument("--quant")
    c.add_argument("--refit", action="store_true", help="refit the constants from the built-in measurements")
    c.add_argument("--out", help="with --refit: write the calibration JSON here")
    c.add_argument("--role", choices=["fit", "backtest"], help="only show fit or held-out rows")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_calibrate)

    v = sub.add_parser("vram", help="the upstream web calculator's VRAM rule of thumb")
    v.add_argument("--params", type=float, required=True, help="parameters in billions")
    v.add_argument("--quant", default="Q4", choices=U.MODEL_QUANTS)
    v.add_argument("--context", type=int, default=4096)
    v.add_argument("--no-kv-cache", action="store_true")
    v.add_argument("--kv-quant", default="F16", choices=U.KV_QUANTS)
    v.add_argument("--mode", default="incremental", choices=U.MODES)
    v.add_argument("--memory-mode", default="DISCRETE_GPU", choices=U.MEMORY_MODES)
    v.add_argument("--system-memory", type=float, default=32)
    v.add_argument("--gpu-vram", type=float, default=24)
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_vram)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (KeyError, ValueError, OSError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        print(f"error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
