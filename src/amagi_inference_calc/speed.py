"""Speed model: decode and prefill tok/s from memory bandwidth and FLOPs.

Decode is memory-bound. Every step reads the active weights once plus the
KV cache of every stream, so

    step = bytes_read / effective_bandwidth + overhead
           + moe_layer_cost * moe_layers           (MoE routing, gathers)
           + (streams - 1) * batch_cost             (extra work per stream)

For dense models batch_cost is the stream's matmul FLOPs at the prefill
rate (untested: no dense multi-stream runs yet). For MoE models it is a
fixed cost per MoE layer, fitted on Qwen3.6 and held out on gpt-oss.

Prefill is compute-bound:

    prefill_s = 2 * active_params * prompt / F  +  attention_flops / F_dense

The constants come from fit() over data/measurements.json, which holds
real runs on one RTX 3090. Other GPUs scale by memory bandwidth; prefill
is only reported where FLOPs are known (the calibrated GPU or --tflops).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .data import load_json
from .models import ModelSpec, find_model

GIB = 1024 ** 3
RUNTIME_OVERHEAD_GIB = 0.5  # CUDA context + compute buffers; measured 0.11-0.46 GiB


@dataclass
class Calibration:
    gpu: str
    bw_ref_gbs: float          # bandwidth the fit was scaled against (memtest)
    bw_spec_gbs: float         # vendor spec of the calibration GPU
    bw_eff_gbs: float          # bandwidth decode actually achieved
    overhead_ms: float         # fixed cost per decode step
    moe_layer_ms: float        # extra cost per MoE layer per step
    batch_ms_per_b_dense: float  # extra ms per added stream per 1B active params
    batch_ms_per_layer_moe: float  # extra ms per added stream per MoE layer
    tflops_dense: float        # effective prefill FLOP/s (dense matmuls), in TFLOPS
    tflops_moe: float          # effective prefill FLOP/s for MoE expert matmuls
    notes: List[str] = field(default_factory=list)

    @property
    def decode_eff(self) -> float:
        """Achieved decode bandwidth as a share of the vendor spec."""
        return self.bw_eff_gbs / self.bw_spec_gbs


# -- GPUs --------------------------------------------------------------------

@dataclass
class GPU:
    id: str
    name: str
    vendor: str
    kind: str
    vram_gb: float
    bw_spec_gbs: float
    bw_measured_gbs: Optional[float] = None
    price_used_usd: Optional[float] = None
    price_source: Optional[str] = None
    memory_gb: Optional[float] = None
    tflops: Optional[float] = None  # effective prefill TFLOPS, if known
    bw_decode_gbs: Optional[float] = None  # decode bandwidth fitted from your own run
    vram_note: Optional[str] = None
    note: Optional[str] = None

    @property
    def vram_bytes(self) -> float:
        return self.vram_gb * GIB


def builtin_gpus() -> List[GPU]:
    fields = GPU.__dataclass_fields__
    return [GPU(**{k: v for k, v in g.items() if k in fields}) for g in load_json("gpus.json")["gpus"]]


def find_gpu(key: str) -> GPU:
    gpus = builtin_gpus()
    k = key.lower().replace(" ", "-")
    for g in gpus:
        if g.id == k:
            return g
    hits = sorted((g for g in gpus if k in g.id or key.lower() in g.name.lower()), key=lambda g: len(g.id))
    if len(hits) == 1 or (hits and all(h.id.startswith(hits[0].id) for h in hits)):
        return hits[0]  # "3090" means rtx-3090, not rtx-3090-ti
    if hits:
        raise KeyError(f"{key!r} matches several GPUs: {', '.join(g.id for g in hits)}")
    raise KeyError(f"unknown GPU {key!r}; run `amagi-inference-calc gpus` for the list")


def custom_gpu(vram_gb: float, bw_gbs: float, name: str = "custom GPU", price: Optional[float] = None) -> GPU:
    return GPU(id="custom", name=name, vendor="custom", kind="discrete", vram_gb=vram_gb,
               bw_spec_gbs=bw_gbs, price_used_usd=price)


# -- calibration -------------------------------------------------------------

def load_calibration() -> Calibration:
    c = load_json("calibration.json")
    fields = Calibration.__dataclass_fields__
    return Calibration(**{k: v for k, v in c.items() if k in fields})


def _bandwidth(gpu: GPU, cal: Calibration) -> Tuple[float, str]:
    """Effective decode bandwidth (bytes/s) and where it came from."""
    if gpu.bw_decode_gbs:
        return gpu.bw_decode_gbs * 1e9, "fitted from your benchmark run"
    if gpu.id == cal.gpu:
        return cal.bw_eff_gbs * 1e9, "calibrated on this GPU"
    if gpu.bw_measured_gbs:
        scale = cal.bw_eff_gbs / cal.bw_ref_gbs
        return gpu.bw_measured_gbs * scale * 1e9, f"memtest {gpu.bw_measured_gbs:g} GB/s x {scale:.2f}"
    return gpu.bw_spec_gbs * cal.decode_eff * 1e9, f"spec x {cal.decode_eff:.2f}"


def _moe_layers(m: ModelSpec) -> int:
    return m.layers if m.experts_total else 0


def attention_flops(m: ModelSpec, prompt: int) -> float:
    """FLOPs for QK^T and AV over a causal prompt (full + sliding layers)."""
    width = m.attn_heads * (m.head_dim_k + m.head_dim_v) / 2
    full = 2.0 * prompt * prompt * width * m.full_attn_layers
    w = min(prompt, m.window) if m.window else prompt
    windowed = 4.0 * prompt * w * width * m.window_attn_layers * (0.5 if w == prompt else 1.0)
    return full + windowed


def decode_step_s(m: ModelSpec, quant: str, context: int, streams: int, bw: float,
                  cal: Calibration, kv_type: str = "F16") -> float:
    mem = m.read_bytes_per_step(quant, streams) + streams * m.kv_bytes(context, kv_type)
    t = mem / bw + cal.overhead_ms / 1e3 + cal.moe_layer_ms / 1e3 * _moe_layers(m)
    if m.experts_total:
        per_stream = cal.batch_ms_per_layer_moe * _moe_layers(m)
    else:
        per_stream = cal.batch_ms_per_b_dense * m.active_params / 1e9
    t += max(0, streams - 1) * per_stream / 1e3
    return t


def prefill_s(m: ModelSpec, prompt: int, tflops_dense: float, tflops_moe: Optional[float] = None) -> float:
    lin_f = (tflops_moe if (m.experts_total and tflops_moe) else tflops_dense) * 1e12
    return 2.0 * m.active_params * prompt / lin_f + attention_flops(m, prompt) / (tflops_dense * 1e12)


# -- the estimate --------------------------------------------------------------

@dataclass
class Estimate:
    model: str
    quant: str
    gpu: str
    gpu_count: int
    users: int
    context: int
    kv_type: str
    weights_gib: float
    kv_gib: float
    vram_needed_gib: float
    vram_total_gib: float
    fits: bool
    cards_needed: int
    bandwidth_gbs: float
    bandwidth_source: str
    tok_s_single: float
    tok_s_per_user: float
    tok_s_total: float
    prefill_tok_s: Optional[float]
    ttft_s: Optional[float]
    ttft_all_s: Optional[float]
    prompt: int
    price_usd: Optional[float]
    usd_per_tok_s: Optional[float]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        out = dict(self.__dict__)
        for k, v in list(out.items()):
            if isinstance(v, float):
                out[k] = round(v, 3)
        return out


def estimate(
    model: ModelSpec,
    gpu: GPU,
    quant: Optional[str] = None,
    users: int = 1,
    context: int = 8192,
    prompt: Optional[int] = None,
    gpu_count: int = 1,
    kv_type: str = "F16",
    tflops: Optional[float] = None,
    price: Optional[float] = None,
    cal: Optional[Calibration] = None,
) -> Estimate:
    """Estimate one model on `gpu_count` copies of `gpu` serving `users` at once.

    `context` is the context window each user gets (KV is reserved for all
    of it); `prompt` is the prompt length used for the speed and TTFT
    numbers (defaults to half the context, as a mid-conversation point).
    """
    cal = cal or load_calibration()
    quant = quant or model.native_quant or "Q4_K_M"
    users = max(1, int(users))
    gpu_count = max(1, int(gpu_count))
    prompt = context // 2 if prompt is None else min(prompt, context)
    warnings: List[str] = []

    weights = model.weight_bytes(quant)
    kv = users * model.kv_bytes(context, kv_type)
    need = weights + kv + RUNTIME_OVERHEAD_GIB * GIB * gpu_count
    total = gpu.vram_bytes * gpu_count
    fits = need <= total
    per_card = gpu.vram_bytes - RUNTIME_OVERHEAD_GIB * GIB
    cards = max(1, math.ceil((weights + kv) / per_card)) if per_card > 0 else 0
    if model.context_max and context > model.context_max:
        warnings.append(f"context {context} is above the model's trained {model.context_max}")
    if not fits:
        warnings.append(f"does not fit: needs {need / GIB:.1f} GiB, has {total / GIB:.1f} GiB ({cards} card(s) needed)")
    if gpu_count > 1:
        warnings.append("multi-GPU assumes llama.cpp layer split: memory adds up, speed does not (cards take turns)")
    if gpu.note:
        warnings.append(gpu.note)
    if gpu.vendor in ("amd", "intel", "apple"):
        warnings.append(f"{gpu.vendor.upper()} speed is scaled from an NVIDIA/CUDA calibration; real numbers often land lower")

    bw, bw_src = _bandwidth(gpu, cal)
    single = 1.0 / decode_step_s(model, quant, prompt, 1, bw, cal, kv_type)
    step = decode_step_s(model, quant, prompt, users, bw, cal, kv_type)
    per_user = 1.0 / step

    flops = tflops if tflops else (cal.tflops_dense if gpu.id == cal.gpu else gpu.tflops)
    pp = ttft = ttft_all = None
    if flops:
        f_moe = cal.tflops_moe * (flops / cal.tflops_dense)
        t_pp = prefill_s(model, prompt, flops, f_moe)
        pp = prompt / t_pp if t_pp > 0 else None
        ttft = t_pp
        ttft_all = t_pp * users

    if price is None:
        price = gpu.price_used_usd
    total_price = price * gpu_count if price is not None else None
    usd = total_price / (per_user * users) if (total_price and fits) else None
    if users > 8:
        warnings.append("more than 8 users is beyond the measured range (1-8 streams)")

    return Estimate(
        model=model.id, quant=quant, gpu=gpu.id, gpu_count=gpu_count, users=users,
        context=context, kv_type=kv_type,
        weights_gib=weights / GIB, kv_gib=kv / GIB, vram_needed_gib=need / GIB,
        vram_total_gib=total / GIB, fits=fits, cards_needed=cards,
        bandwidth_gbs=bw / 1e9, bandwidth_source=bw_src,
        tok_s_single=single, tok_s_per_user=per_user, tok_s_total=per_user * users,
        prefill_tok_s=pp, ttft_s=ttft, ttft_all_s=ttft_all, prompt=prompt,
        price_usd=total_price, usd_per_tok_s=usd, warnings=warnings,
    )


# -- fitting -----------------------------------------------------------------

def _lstsq2(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Least squares y = a*x + b."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    return a, my - a * mx


def fit(points: Optional[List[dict]] = None, gpu_id: str = "rtx-3090",
        bw_ref_gbs: float = 737.8, bw_spec_gbs: float = 936.0) -> Calibration:
    """Fit the calibration constants from measured runs on one GPU.

    1. Dense single-stream decode -> effective bandwidth and fixed overhead.
    2. MoE single-stream decode   -> extra ms per MoE layer.
    3. MoE multi-stream decode    -> extra ms per added stream per MoE layer.
    4. Dense prefill              -> effective dense TFLOPS.
    5. MoE prefill                -> effective expert-matmul TFLOPS.
    Only rows with role "fit" are used.
    """
    data = load_json("measurements.json")
    points = [p for p in (points or data["points"]) if p.get("role", "fit") == "fit"]
    kv_type = data.get("kv_type", "F16")
    specs = {}

    def spec(mid: str) -> ModelSpec:
        if mid not in specs:
            specs[mid] = find_model(mid)
        return specs[mid]

    dec = [p for p in points if p["kind"] == "decode" and p.get("streams", 1) == 1]
    dense = [p for p in dec if not spec(p["model"]).experts_total]
    xs = [spec(p["model"]).read_bytes_per_step(p["quant"]) + spec(p["model"]).kv_bytes(p["context"], kv_type) for p in dense]
    ys = [1.0 / p["tok_s"] for p in dense]
    a, o = _lstsq2(xs, ys)
    bw = 1.0 / a

    base = Calibration(gpu=gpu_id, bw_ref_gbs=bw_ref_gbs, bw_spec_gbs=bw_spec_gbs,
                       bw_eff_gbs=bw / 1e9, overhead_ms=o * 1e3, moe_layer_ms=0.0,
                       batch_ms_per_b_dense=0.0, batch_ms_per_layer_moe=0.0,
                       tflops_dense=0.0, tflops_moe=0.0)

    moe = [p for p in dec if spec(p["model"]).experts_total]
    extra = []
    for p in moe:
        m = spec(p["model"])
        t = decode_step_s(m, p["quant"], p["context"], 1, bw, base, kv_type)
        extra.append((1.0 / p["tok_s"] - t) * 1e3 / _moe_layers(m))
    base.moe_layer_ms = sum(extra) / len(extra) if extra else 0.0

    pp = [p for p in points if p["kind"] == "prefill"]
    pd = [p for p in pp if not spec(p["model"]).experts_total]
    f_dense = [(2 * spec(p["model"]).active_params * p["prompt"] + attention_flops(spec(p["model"]), p["prompt"]))
               / (p["prompt"] / p["tok_s"]) for p in pd]
    base.tflops_dense = sum(f_dense) / len(f_dense) / 1e12 if f_dense else 0.0
    # dense batching: one more stream costs its matmul FLOPs at the prefill rate
    if base.tflops_dense:
        base.batch_ms_per_b_dense = 2e9 / (base.tflops_dense * 1e12) * 1e3

    pm = [p for p in pp if spec(p["model"]).experts_total]
    f_moe = []
    for p in pm:
        m = spec(p["model"])
        t = p["prompt"] / p["tok_s"] - attention_flops(m, p["prompt"]) / (base.tflops_dense * 1e12)
        f_moe.append(2 * m.active_params * p["prompt"] / t)
    if f_moe:  # geometric mean: the MoE rows differ a lot
        base.tflops_moe = math.exp(sum(math.log(f) for f in f_moe) / len(f_moe)) / 1e12

    bat = [p for p in points if p["kind"] == "decode" and p.get("streams", 1) > 1]
    per_b = []
    for p in bat:
        m = spec(p["model"])
        t0 = decode_step_s(m, p["quant"], p["context"], p["streams"], bw, base, kv_type)
        resid = 1.0 / p["tok_s"] - t0
        per_b.append(resid * 1e3 / ((p["streams"] - 1) * _moe_layers(m)))
    if per_b:
        base.batch_ms_per_layer_moe = sum(per_b) / len(per_b)
    return base


def backtest(cal: Optional[Calibration] = None, role: Optional[str] = None) -> List[Dict[str, object]]:
    """Predicted vs measured for every row in measurements.json."""
    cal = cal or load_calibration()
    data = load_json("measurements.json")
    kv_type = data.get("kv_type", "F16")
    bw = cal.bw_eff_gbs * 1e9
    rows = []
    for p in data["points"]:
        if role and p.get("role") != role:
            continue
        m = find_model(p["model"])
        if p["kind"] == "decode":
            n = p.get("streams", 1)
            pred = 1.0 / decode_step_s(m, p["quant"], p["context"], n, bw, cal, kv_type)
            where = f"ctx {p['context']}, {n} stream(s)"
        else:
            pred = p["prompt"] / prefill_s(m, p["prompt"], cal.tflops_dense, cal.tflops_moe)
            where = f"prompt {p['prompt']}"
        rows.append({
            "kind": p["kind"], "model": p["model"], "where": where, "role": p.get("role"),
            "measured": p["tok_s"], "predicted": round(pred, 1),
            "error_pct": round((pred - p["tok_s"]) / p["tok_s"] * 100, 1),
        })
    return rows


def decode_bw_from_runs(model: ModelSpec, quant: str, runs: Sequence[Tuple[int, float]],
                        cal: Optional[Calibration] = None, kv_type: str = "F16") -> float:
    """Effective decode bandwidth (GB/s) that explains measured single-user runs.

    `runs` is a list of (context_tokens, decode_tok_s). Each run is solved
    for bandwidth with the calibrated step overhead held fixed; the median
    is returned.
    """
    cal = cal or load_calibration()
    fixed = cal.overhead_ms / 1e3 + cal.moe_layer_ms / 1e3 * _moe_layers(model)
    vals = []
    for ctx, tok_s in runs:
        mem = model.read_bytes_per_step(quant) + model.kv_bytes(ctx, kv_type)
        t = 1.0 / tok_s - fixed
        if t > 0:
            vals.append(mem / t / 1e9)
    if not vals:
        raise ValueError("no usable runs (decode faster than the fixed step overhead?)")
    vals.sort()
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def tokbench_runs(report: dict) -> List[Tuple[int, float]]:
    """(context, decode tok/s) pairs from an amagi-tokbench JSON report."""
    out = []
    for case in report.get("cases", []):
        med = case.get("median") or {}
        tps = med.get("server_gen_tps") or med.get("gen_tps")
        if not tps or case.get("concurrency", 1) not in (1, None):
            continue
        ctx = int(med.get("prompt_tokens", 0) + med.get("gen_tokens", 0) / 2)
        out.append((ctx, float(tps)))
    return out
