"""Model specs: the handful of architecture numbers the calculator needs.

A spec can come from three places: the built-in list (data/models.json),
a Hugging Face config.json, or the header of a GGUF file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import quant as Q
from .data import load_json


@dataclass
class ModelSpec:
    id: str
    name: str
    params: int                      # every weight in the file
    embed_params: int = 0            # input token embedding (read one row per token)
    tied_embeddings: bool = False    # embedding doubles as the output head
    expert_params: int = 0           # all routed-expert weights
    experts_total: int = 0
    experts_used: int = 0
    layers: int = 0
    full_attn_layers: int = 0        # attention layers whose KV grows with context
    window_attn_layers: int = 0      # sliding-window layers (KV capped at `window`)
    window: int = 0
    kv_heads: int = 0
    head_dim_k: int = 0
    head_dim_v: int = 0
    context_max: int = 0
    native_quant: str = ""           # e.g. MXFP4 for gpt-oss; empty means "pick one"
    source: str = ""
    # exact numbers for one specific file (GGUF); used when quant matches
    file_quant: str = ""
    file_bytes: int = 0
    read_bytes: int = 0
    notes: List[str] = field(default_factory=list)

    # -- weights ---------------------------------------------------------
    @property
    def active_params(self) -> int:
        """Weights a single decode step multiplies (no input embedding)."""
        frac = self.experts_used / self.experts_total if self.experts_total else 1.0
        dense = self.params - self.expert_params
        if not self.tied_embeddings:
            dense -= self.embed_params
        return int(dense + self.expert_params * frac)

    def _bpw(self, quant: str) -> float:
        return Q.bits_per_weight(quant)

    def _exact(self, quant: str) -> bool:
        if not self.file_quant:
            return False
        try:
            return Q.normalize(quant) == Q.normalize(self.file_quant)
        except KeyError:
            return False

    def weight_bytes(self, quant: str) -> float:
        if self._exact(quant) and self.file_bytes:
            return float(self.file_bytes)
        return self.params * self._bpw(quant) / 8

    def expert_union(self, streams: int) -> float:
        """Expected share of experts touched in one step by `streams` tokens."""
        if not self.experts_total:
            return 1.0
        p = self.experts_used / self.experts_total
        return 1.0 - (1.0 - p) ** max(1, streams)

    def read_bytes_per_step(self, quant: str, streams: int = 1) -> float:
        """Weight bytes one decode step reads for `streams` parallel tokens."""
        if self._exact(quant) and self.read_bytes:
            # scale the exact single-stream number for the expert union
            if streams > 1 and self.experts_total:
                one = self.expert_union(1)
                exp_one = self.expert_params * self._bpw(quant) / 8 * one
                return self.read_bytes - exp_one + exp_one / one * self.expert_union(streams)
            return float(self.read_bytes)
        bpw = self._bpw(quant) / 8
        dense = self.params - self.expert_params
        if not self.tied_embeddings:
            dense -= self.embed_params
        return (dense + self.expert_params * self.expert_union(streams)) * bpw

    # -- KV cache ----------------------------------------------------------
    def kv_bytes(self, context: int, kv_type: str = "F16") -> float:
        """KV cache bytes for one sequence of `context` tokens."""
        per_layer_tok = self.kv_heads * (self.head_dim_k + self.head_dim_v) * Q.kv_bytes(kv_type)
        full = self.full_attn_layers * per_layer_tok * context
        windowed = self.window_attn_layers * per_layer_tok * min(context, self.window or context)
        return full + windowed

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["active_params"] = self.active_params
        return d


def _from_dict(d: Dict[str, Any]) -> ModelSpec:
    fields = ModelSpec.__dataclass_fields__
    return ModelSpec(**{k: v for k, v in d.items() if k in fields})


def builtin_models() -> List[ModelSpec]:
    return [_from_dict(m) for m in load_json("models.json")["models"]]


def find_model(key: str) -> ModelSpec:
    """Look up a built-in model by id (exact, then substring), or load a file."""
    if os.path.isfile(key):
        return load_model_file(key)
    models = builtin_models()
    k = key.lower()
    for m in models:
        if m.id.lower() == k:
            return m
    hits = [m for m in models if k in m.id.lower() or k in m.name.lower()]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise KeyError(f"{key!r} matches several models: {', '.join(m.id for m in hits)}")
    raise KeyError(f"unknown model {key!r}; run `amagi-inference-calc models` for the list")


def load_model_file(path: str) -> ModelSpec:
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head == b"GGUF":
        from .gguf import read_gguf
        return from_gguf(read_gguf(path))
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if "params" in data and "id" in data:
        return _from_dict(data)
    return from_hf_config(data, model_id=os.path.splitext(os.path.basename(path))[0])


# -- Hugging Face config.json ------------------------------------------------

def _cfg(c: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if c.get(k) is not None:
            return c[k]
    return default


def from_hf_config(config: Dict[str, Any], model_id: str = "custom", name: str = "") -> ModelSpec:
    """Build a spec from a transformers config.json.

    Parameter counts are computed from the layer shapes (attention,
    MLP or routed experts, embeddings). Norms and biases are left out,
    which costs well under 0.1%. Vision towers are not counted.
    """
    c = config.get("text_config", config)
    hidden = int(_cfg(c, "hidden_size", "n_embd", "d_model"))
    layers = int(_cfg(c, "num_hidden_layers", "n_layer", "num_layers"))
    heads = int(_cfg(c, "num_attention_heads", "n_head"))
    kv_heads = int(_cfg(c, "num_key_value_heads", default=heads))
    head_dim = int(_cfg(c, "head_dim", default=hidden // heads))
    vocab = int(_cfg(c, "vocab_size"))
    inter = int(_cfg(c, "intermediate_size", default=4 * hidden))
    tied = bool(_cfg(c, "tie_word_embeddings", default=config.get("tie_word_embeddings", True)))  # transformers default
    n_exp = int(_cfg(c, "num_experts", "num_local_experts", "n_routed_experts", default=0) or 0)
    n_used = int(_cfg(c, "num_experts_per_tok", "num_experts_per_token", default=0) or 0)
    moe_inter = int(_cfg(c, "moe_intermediate_size", default=inter))
    n_shared = int(_cfg(c, "n_shared_experts", "num_shared_experts", default=0) or 0)

    attn = hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim + heads * head_dim * hidden
    if n_exp:
        experts = n_exp * 3 * hidden * moe_inter
        dense_mlp = n_shared * 3 * hidden * moe_inter + hidden * n_exp  # shared experts + router
    else:
        experts = 0
        dense_mlp = 3 * hidden * inter
    embed = vocab * hidden
    params = layers * (attn + dense_mlp + experts) + embed + (0 if tied else embed)

    # attention layout: full vs sliding window
    window = int(_cfg(c, "sliding_window", default=0) or 0)
    layer_types = c.get("layer_types")
    use_sw = c.get("use_sliding_window", None)
    pattern = int(_cfg(c, "sliding_window_pattern", default=0) or 0)
    max_pos = int(_cfg(c, "max_position_embeddings", default=0) or 0)
    if isinstance(layer_types, list) and layer_types:
        full = sum(1 for t in layer_types if "full" in str(t))
        win = sum(1 for t in layer_types if "sliding" in str(t))
    elif pattern and window:
        full = layers // pattern
        win = layers - full
    else:
        full, win = layers, 0
        if use_sw is False or (window and max_pos and window >= max_pos):
            window = 0
        elif window and use_sw:
            full, win = 0, layers
    if not win:
        window = 0

    return ModelSpec(
        id=model_id,
        name=name or str(config.get("_name_or_path") or model_id),
        params=int(params),
        embed_params=int(embed),
        tied_embeddings=tied,
        expert_params=int(layers * experts),
        experts_total=n_exp,
        experts_used=n_used,
        layers=layers,
        full_attn_layers=full,
        window_attn_layers=win,
        window=window,
        kv_heads=kv_heads,
        head_dim_k=head_dim,
        head_dim_v=head_dim,
        context_max=max_pos,
        source="config.json",
    )


# -- GGUF --------------------------------------------------------------------

def from_gguf(info: Any) -> ModelSpec:
    """Build a spec from a parsed GGUF header (gguf.GGUFInfo)."""
    names = {t.name: t for t in info.tensors}
    tied = "output.weight" not in names
    embed = names["token_embd.weight"].n_elements if "token_embd.weight" in names else 0
    mtp = tuple(f"blk.{i}." for i in info.mtp_layers())
    expert_params = sum(t.n_elements for t in info.tensors if "_exps." in t.name and not (mtp and t.name.startswith(mtp)))
    mtp_params = sum(t.n_elements for t in info.tensors if mtp and t.name.startswith(mtp))
    n_head = int(info.arch_value("attention.head_count", 0) or 0) or 1
    emb = int(info.arch_value("embedding_length", 0) or 0)
    k_len = int(info.arch_value("attention.key_length", 0) or 0) or (emb // n_head if emb else 0)
    v_len = int(info.arch_value("attention.value_length", 0) or 0) or k_len
    kv = info.arch_value("attention.head_count_kv", n_head)
    if isinstance(kv, list):
        kv_layers = info.kv_layers()
        vals = [int(kv[i]) for i in kv_layers if i < len(kv) and int(kv[i]) > 0]
        kv = max(vals) if vals else n_head
    window = int(info.arch_value("attention.sliding_window", 0) or 0)
    kv_layers = len(info.kv_layers())
    notes = []
    if mtp:
        notes.append(f"{len(mtp)} MTP (nextn) block(s) excluded from per-token reads")
    full, win = kv_layers, 0
    pattern = info.arch_value("attention.sliding_window_pattern")
    if window and isinstance(pattern, list) and pattern:
        win = sum(1 for i in info.kv_layers() if i < len(pattern) and pattern[i])
        full = kv_layers - win
    elif window and isinstance(pattern, int) and pattern > 1:
        full = kv_layers // pattern
        win = kv_layers - full
    stem = os.path.splitext(os.path.basename(info.path))[0]
    return ModelSpec(
        id=stem.lower(),
        name=info.name,
        params=int(info.n_params - mtp_params),
        embed_params=int(embed),
        tied_embeddings=tied,
        expert_params=int(expert_params),
        experts_total=info.expert_count,
        experts_used=info.expert_used_count,
        layers=info.n_layers - len(mtp),
        full_attn_layers=full,
        window_attn_layers=win,
        window=window if win else 0,
        kv_heads=int(kv),
        head_dim_k=k_len,
        head_dim_v=v_len,
        context_max=int(info.arch_value("context_length", 0) or 0),
        source=f"GGUF header ({os.path.basename(info.path)})",
        file_quant=info.quant,
        file_bytes=int(info.file_bytes),
        read_bytes=int(info.bytes_read_per_token()),
        notes=notes,
    )
