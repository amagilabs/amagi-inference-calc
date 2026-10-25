"""Bits per weight for common quantization formats.

Pure ggml block types have an exact size: a Q4_K super-block packs 256
weights into 144 bytes, which is 4.5 bits per weight. The "_M"/"_S" mixes
keep some tensors (attention V, FFN down, the output head) at a higher type,
so they land above the block size. For Q4_K_M we use the value measured on
two real files; the other mixes are marked as estimates.
"""

from __future__ import annotations

from typing import Dict, Tuple

# name -> (bits per weight, how we know)
QUANTS: Dict[str, Tuple[float, str]] = {
    "F32": (32.0, "exact"),
    "F16": (16.0, "exact"),
    "BF16": (16.0, "exact"),
    "Q8_0": (8.5, "exact: 34-byte block per 32 weights"),
    "Q6_K": (6.5625, "exact: 210-byte block per 256 weights"),
    "Q5_K_M": (5.75, "estimate: Q5_K block (5.5) plus mixed tensors"),
    "Q5_K_S": (5.54, "estimate: Q5_K block (5.5) plus mixed tensors"),
    "Q5_0": (5.5, "exact: 22-byte block per 32 weights"),
    "Q4_K_M": (4.88, "measured: Qwen2.5-7B 4.91, Qwen2.5-32B 4.85 bits/weight"),
    "Q4_K_S": (4.58, "estimate: Q4_K block (4.5) plus mixed tensors"),
    "Q4_0": (4.5, "exact: 18-byte block per 32 weights"),
    "IQ4_XS": (4.25, "exact: 136-byte block per 256 weights"),
    "MXFP4": (4.25, "exact: 17-byte block per 32 weights"),
    "Q3_K_M": (3.91, "estimate: Q3_K block (3.4375) plus mixed tensors"),
    "Q3_K_S": (3.5, "estimate: Q3_K block (3.4375) plus mixed tensors"),
    "Q2_K": (3.0, "estimate: Q2_K block (2.625) plus mixed tensors"),
    # GPU-side formats used by vLLM / ExLlama
    "AWQ": (4.25, "estimate: 4-bit weights, group 128 scales and zeros"),
    "GPTQ": (4.25, "estimate: 4-bit weights, group 128 scales and zeros"),
    "FP8": (8.0, "exact"),
}

# KV cache element sizes (bytes per element)
KV_TYPES: Dict[str, float] = {
    "F32": 4.0,
    "F16": 2.0,
    "BF16": 2.0,
    "Q8_0": 34 / 32,
    "Q5_1": 24 / 32,
    "Q4_0": 18 / 32,
    "FP8": 1.0,
}

_ALIASES = {
    "Q4": "Q4_K_M", "Q5": "Q5_K_M", "Q6": "Q6_K", "Q8": "Q8_0", "Q3": "Q3_K_M",
    "Q2": "Q2_K", "FP16": "F16", "FP32": "F32", "INT8": "Q8_0", "INT4": "Q4_K_M",
    "Q4_K": "Q4_K_M", "Q5_K": "Q5_K_M", "Q3_K": "Q3_K_M", "MXFP4_MOE": "MXFP4",
}


def normalize(name: str) -> str:
    key = name.strip().upper().replace("-", "_")
    if key.startswith("UD_"):
        key = key[3:]
    key = _ALIASES.get(key, key)
    if key not in QUANTS:
        raise KeyError(f"unknown quant {name!r}; known: {', '.join(sorted(QUANTS))}")
    return key


def bits_per_weight(name: str) -> float:
    return QUANTS[normalize(name)][0]


def kv_bytes(name: str) -> float:
    key = name.strip().upper().replace("-", "_")
    key = {"Q8": "Q8_0", "Q4": "Q4_0", "FP16": "F16", "Q5": "Q5_1"}.get(key, key)
    if key not in KV_TYPES:
        raise KeyError(f"unknown KV cache type {name!r}; known: {', '.join(sorted(KV_TYPES))}")
    return KV_TYPES[key]
