"""Read the header of a GGUF model file (stdlib only).

Only the metadata and tensor table are read, never the weights, so this is
fast even for a 40 GB file. Tensor sizes come from the gap between tensor
offsets, which is exact for every quant type without a block-size table.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional

GGUF_MAGIC = b"GGUF"

# value types from the GGUF spec
_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_STRING, _ARRAY = 8, 9

# ggml tensor type ids -> names (display only; sizes come from offsets)
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}

# general.file_type ids (llama_ftype) -> the quant name people use
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE",
}

_BLOCK = re.compile(r"^blk\.(\d+)\.")


class GGUFError(ValueError):
    """The file is not a GGUF file or its header is damaged."""


@dataclass
class Tensor:
    name: str
    dims: List[int]
    type_id: int
    offset: int
    nbytes: int = 0

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass
class GGUFInfo:
    path: str
    file_bytes: int
    version: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensors: List[Tensor] = field(default_factory=list)

    # -- generic helpers -------------------------------------------------
    @property
    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    def arch_value(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(f"{self.arch}.{key}", default)

    @property
    def name(self) -> str:
        return str(self.metadata.get("general.name") or os.path.basename(self.path))

    @property
    def quant(self) -> str:
        ft = self.metadata.get("general.file_type")
        if isinstance(ft, int) and ft in FILE_TYPES:
            return FILE_TYPES[ft]
        m = re.search(r"(I?Q\d_[A-Z0-9_]+|Q\d_\d|BF16|F16|F32)", os.path.basename(self.path), re.I)
        return m.group(1).upper() if m else "unknown"

    @property
    def n_params(self) -> int:
        return sum(t.n_elements for t in self.tensors)

    @property
    def weight_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    # -- architecture facts used by the calculator -----------------------
    @property
    def n_layers(self) -> int:
        return int(self.arch_value("block_count", 0) or 0)

    @property
    def expert_count(self) -> int:
        return int(self.arch_value("expert_count", 0) or 0)

    @property
    def expert_used_count(self) -> int:
        return int(self.arch_value("expert_used_count", 0) or 0)

    def _per_layer(self, key: str, default: int = 0) -> List[int]:
        v = self.arch_value(key, default)
        if isinstance(v, list):
            vals = [int(x) for x in v]
            return (vals + [vals[-1] if vals else default] * self.n_layers)[: self.n_layers]
        return [int(v or default)] * self.n_layers

    def _blocks(self) -> Dict[int, List[str]]:
        blocks: Dict[int, List[str]] = {}
        for t in self.tensors:
            m = _BLOCK.match(t.name)
            if m:
                blocks.setdefault(int(m.group(1)), []).append(t.name[m.end():])
        return blocks

    def mtp_layers(self) -> List[int]:
        """Multi-token-prediction (nextn) blocks: not run in normal decoding."""
        return sorted(i for i, names in self._blocks().items() if any(n.startswith("nextn.") for n in names))

    def kv_layers(self) -> List[int]:
        """Block indexes that hold a regular attention KV cache.

        Hybrid models mix linear-attention (SSM / DeltaNet) blocks with
        attention blocks, and only the attention blocks keep a KV cache. A
        block counts when it has an attn_k tensor, or a fused attn_qkv
        tensor with no ssm_* tensors next to it. MTP blocks are skipped.
        Falls back to every block when the tensor names are unfamiliar.
        """
        mtp = set(self.mtp_layers())
        layers = []
        for i, names in sorted(self._blocks().items()):
            if i in mtp:
                continue
            has_ssm = any(n.startswith("ssm_") for n in names)
            if "attn_k.weight" in names or ("attn_qkv.weight" in names and not has_ssm):
                layers.append(i)
        if layers:
            return layers
        return [i for i in range(self.n_layers) if i not in mtp]

    def kv_bytes_per_token(self, kv_bytes_per_elem: float = 2.0) -> float:
        """KV cache bytes for one token of context (all attention layers)."""
        heads_kv = self._per_layer("attention.head_count_kv", int(self.arch_value("attention.head_count", 0) or 0))
        n_head = int(self.arch_value("attention.head_count", 0) or 0) or 1
        emb = int(self.arch_value("embedding_length", 0) or 0)
        k_len = int(self.arch_value("attention.key_length", 0) or 0) or (emb // n_head if emb else 0)
        v_len = int(self.arch_value("attention.value_length", 0) or 0) or k_len
        total = 0.0
        for i in self.kv_layers():
            h = heads_kv[i] if i < len(heads_kv) else (heads_kv[-1] if heads_kv else 0)
            total += h * (k_len + v_len) * kv_bytes_per_elem
        return total

    def bytes_read_per_token(self) -> int:
        """Weight bytes a single decode step reads.

        Every tensor is read once per token, except: the token embedding
        table (only one row is looked up, unless it is tied to the output
        head) and MoE expert tensors (only expert_used/expert_count of them).
        """
        names = {t.name for t in self.tensors}
        tied = "output.weight" not in names
        frac = (self.expert_used_count / self.expert_count) if self.expert_count else 1.0
        mtp = tuple(f"blk.{i}." for i in self.mtp_layers())
        total = 0.0
        for t in self.tensors:
            if mtp and t.name.startswith(mtp):
                continue
            if t.name == "token_embd.weight" and not tied:
                continue
            if "_exps." in t.name:
                total += t.nbytes * frac
            else:
                total += t.nbytes
        return int(total)

    def active_params(self) -> int:
        frac = (self.expert_used_count / self.expert_count) if self.expert_count else 1.0
        mtp = tuple(f"blk.{i}." for i in self.mtp_layers())
        total = 0.0
        for t in self.tensors:
            if t.name == "token_embd.weight" or (mtp and t.name.startswith(mtp)):
                continue
            total += t.n_elements * (frac if "_exps." in t.name else 1.0)
        return int(total)

    def summary(self) -> Dict[str, Any]:
        return {
            "file": os.path.basename(self.path),
            "name": self.name,
            "arch": self.arch,
            "quant": self.quant,
            "file_gb": round(self.file_bytes / 1e9, 3),
            "params_b": round(self.n_params / 1e9, 3),
            "active_params_b": round(self.active_params() / 1e9, 3),
            "layers": self.n_layers,
            "kv_layers": len(self.kv_layers()),
            "experts": f"{self.expert_used_count}/{self.expert_count}" if self.expert_count else None,
            "context_trained": self.arch_value("context_length"),
            "kv_bytes_per_token_f16": int(self.kv_bytes_per_token(2.0)),
            "mtp_layers": len(self.mtp_layers()),
            "bytes_read_per_token_gb": round(self.bytes_read_per_token() / 1e9, 3),
        }


def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    buf = f.read(size)
    if len(buf) != size:
        raise GGUFError("unexpected end of file in GGUF header")
    return struct.unpack(fmt, buf)[0]


def _read_str(f: BinaryIO) -> str:
    n = _read(f, "<Q")
    if n > 1 << 24:
        raise GGUFError(f"implausible string length {n}")
    buf = f.read(n)
    if len(buf) != n:
        raise GGUFError("unexpected end of file in GGUF string")
    return buf.decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int, keep_arrays: bool) -> Any:
    if vtype in _SCALARS:
        return _read(f, _SCALARS[vtype])
    if vtype == _STRING:
        return _read_str(f)
    if vtype == _ARRAY:
        itype = _read(f, "<I")
        n = _read(f, "<Q")
        if itype in _SCALARS and not keep_arrays:
            # skip big arrays (token lists, scores) without decoding them
            f.seek(struct.calcsize(_SCALARS[itype]) * n, os.SEEK_CUR)
            return None
        if itype == _STRING and n > 64:
            # token and merge lists: skip each string by its length
            for _ in range(n):
                f.seek(_read(f, "<Q"), os.SEEK_CUR)
            return None
        return [_read_value(f, itype, keep_arrays=False) for _ in range(n)]
    raise GGUFError(f"unknown GGUF value type {vtype}")


def read_gguf(path: str) -> GGUFInfo:
    """Parse the GGUF header and tensor table of ``path``."""
    file_bytes = os.path.getsize(path)
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise GGUFError(f"{path}: not a GGUF file (bad magic)")
        version = _read(f, "<I")
        if version < 2:
            raise GGUFError(f"{path}: GGUF version {version} is too old")
        n_tensors = _read(f, "<Q")
        n_kv = _read(f, "<Q")
        meta: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_str(f)
            vtype = _read(f, "<I")
            # per-layer arrays such as head_count_kv are small and needed
            keep = key.endswith(("head_count_kv", "head_count", "sliding_window_pattern"))
            meta[key] = _read_value(f, vtype, keep_arrays=keep)
        tensors: List[Tensor] = []
        for _ in range(n_tensors):
            name = _read_str(f)
            n_dims = _read(f, "<I")
            dims = [_read(f, "<Q") for _ in range(n_dims)]
            ttype = _read(f, "<I")
            offset = _read(f, "<Q")
            tensors.append(Tensor(name, dims, ttype, offset))
        align = int(meta.get("general.alignment", 32) or 32)
        pos = f.tell()
        data_start = pos + (-pos % align)
    order = sorted(tensors, key=lambda t: t.offset)
    data_len = file_bytes - data_start
    for i, t in enumerate(order):
        end = order[i + 1].offset if i + 1 < len(order) else data_len
        t.nbytes = max(0, end - t.offset)
    return GGUFInfo(path=path, file_bytes=file_bytes, version=version, metadata=meta, tensors=tensors)


def write_gguf(path: str, metadata: Dict[str, Any], tensors: List[Dict[str, Any]], align: int = 32) -> None:
    """Write a tiny GGUF file (header + zero-filled tensor data).

    Used by the tests and the demo fixtures; tensors are dicts with
    ``name``, ``dims``, ``type`` (ggml id) and ``nbytes``.
    """

    def w_str(f: BinaryIO, s: str) -> None:
        b = s.encode("utf-8")
        f.write(struct.pack("<Q", len(b)))
        f.write(b)

    def w_val(f: BinaryIO, v: Any) -> None:
        if isinstance(v, bool):
            f.write(struct.pack("<I", 7) + struct.pack("<?", v))
        elif isinstance(v, int):
            f.write(struct.pack("<I", 4 if 0 <= v < 2**32 else 11))
            f.write(struct.pack("<I" if 0 <= v < 2**32 else "<q", v))
        elif isinstance(v, float):
            f.write(struct.pack("<I", 6) + struct.pack("<f", v))
        elif isinstance(v, str):
            f.write(struct.pack("<I", _STRING))
            w_str(f, v)
        elif isinstance(v, list):
            f.write(struct.pack("<I", _ARRAY) + struct.pack("<I", 4) + struct.pack("<Q", len(v)))
            for x in v:
                f.write(struct.pack("<I", int(x)))
        else:
            raise TypeError(f"unsupported metadata value {v!r}")

    with open(path, "wb") as f:
        f.write(GGUF_MAGIC + struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata)))
        for k, v in metadata.items():
            w_str(f, k)
            w_val(f, v)
        offset = 0
        offsets = []
        for t in tensors:
            offsets.append(offset)
            offset += t["nbytes"] + (-t["nbytes"] % align)
        for t, off in zip(tensors, offsets):
            w_str(f, t["name"])
            f.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", t.get("type", 0)) + struct.pack("<Q", off))
        pos = f.tell()
        f.write(b"\0" * (-pos % align))
        f.write(b"\0" * offset)


def read_optional(path: Optional[str]) -> Optional[GGUFInfo]:
    return read_gguf(path) if path else None
