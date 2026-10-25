"""Exact port of the upstream llm-inference-calculator formulas.

Based on web/src/calculations.ts from llm-inference-calculator by Alex
Ziskind (MIT). Kept line-for-line so `amagi-inference-calc vram` gives the
same answer as the upstream web app; tests/test_upstream_parity.py checks it
against the TypeScript running under node. The architecture-aware estimate
lives in memory.py.
"""

from __future__ import annotations

import math
from typing import Dict, Union

MODEL_QUANTS = ("F32", "F16", "Q8", "Q6", "Q5", "Q4", "Q3", "Q2", "GPTQ", "AWQ")
KV_QUANTS = ("F32", "F16", "Q8", "Q5", "Q4")
MODES = ("incremental", "bulk")
MEMORY_MODES = ("DISCRETE_GPU", "UNIFIED_MEMORY")

_MODEL_FACTOR = {
    "F32": 4.0, "F16": 2.0, "Q8": 1.0, "Q6": 0.75, "Q5": 0.625, "Q4": 0.5,
    "Q3": 0.375, "Q2": 0.25, "GPTQ": 0.4, "AWQ": 0.35,
}
_KV_FACTOR = {"F32": 4.0, "F16": 2.0, "Q8": 1.0, "Q5": 0.625, "Q4": 0.5}


def model_quant_factor(q: str) -> float:
    """GB per 1B parameters for a quantization setting."""
    return _MODEL_FACTOR.get(q, 1.0)


def kv_quant_factor(k: str) -> float:
    return _KV_FACTOR.get(k, 1.0)


def required_vram(
    params_b: float,
    model_quant: str,
    context: int,
    use_kv_cache: bool,
    kv_quant: str,
    mode: str = "incremental",
) -> float:
    base = params_b * model_quant_factor(model_quant)
    context_mem = 0.0
    if mode == "incremental":
        if use_kv_cache:
            context_mem = base * 0.2 * (context / 2048) * kv_quant_factor(kv_quant)
    else:
        scale = context / 2048
        context_mem = base * 0.5 * scale
        if use_kv_cache:
            context_mem += base * 0.1 * kv_quant_factor(kv_quant) * scale
    return (base + context_mem) * 1.1


def _js_number(x: float) -> Union[int, float]:
    return int(x) if float(x).is_integer() else x


def hardware_recommendation(
    params_b: float,
    model_quant: str,
    context: int,
    use_kv_cache: bool,
    kv_quant: str,
    memory_mode: str,
    system_memory: float,
    gpu_vram: float,
    mode: str = "incremental",
) -> Dict[str, object]:
    vram = required_vram(params_b, model_quant, context, use_kv_cache, kv_quant, mode)
    base_ram = params_b * model_quant_factor(model_quant) * 0.5 + (context / 1024 if mode == "bulk" else 0)
    system_ram = max(8, base_ram)
    fits_unified = memory_mode == "UNIFIED_MEMORY" and system_memory >= vram
    gpus = 0
    if memory_mode == "DISCRETE_GPU":
        gpus = math.ceil((vram * 1.2) / gpu_vram)
        size = _js_number(gpu_vram)
        if gpus == 1:
            gpu_type = f"Single {size}GB GPU"
        elif gpus <= 8:
            gpu_type = f"{gpus}x {size}GB GPUs"
        else:
            gpu_type = f"Exceeds 8x {size}GB GPUs"
            gpus = 0
    else:
        gpu_type = f"Unified memory ({_js_number(system_memory)}GB)"
    return {
        "gpuType": gpu_type,
        "vramNeeded": f"{vram:.2f}",
        "fitsUnified": fits_unified,
        "systemRamNeeded": _js_number(system_ram),
        "gpusRequired": gpus,
    }


def on_disk_size(params_b: float, model_quant: str) -> float:
    bits = model_quant_factor(model_quant) * 8
    return params_b * 1e9 * bits / 8 / 1e9
