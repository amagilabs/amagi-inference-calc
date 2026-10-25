import pytest

from amagi_inference_calc.gguf import GGUFError, read_gguf
from amagi_inference_calc.models import from_gguf, load_model_file


def test_dense_header(dense_gguf):
    info = read_gguf(str(dense_gguf))
    assert info.quant == "Q4_K_M"
    assert info.n_layers == 4
    assert info.kv_layers() == [0, 1, 2, 3]
    # 1 kv head x (16 + 16) x 2 bytes x 4 layers
    assert info.kv_bytes_per_token() == 1 * 32 * 2 * 4
    # untied: the input embedding is a lookup, not read in full
    embd = next(t for t in info.tensors if t.name == "token_embd.weight")
    assert info.bytes_read_per_token() == info.weight_bytes - embd.nbytes


def test_tensor_sizes_come_from_offsets(dense_gguf):
    info = read_gguf(str(dense_gguf))
    out = next(t for t in info.tensors if t.name == "output.weight")
    assert out.nbytes == 64 * 100 // 2


def test_hybrid_counts_only_attention_layers_and_skips_mtp(hybrid_gguf):
    info = read_gguf(str(hybrid_gguf))
    assert info.mtp_layers() == [6]
    assert info.kv_layers() == [2, 5]
    assert info.kv_bytes_per_token() == 2 * (16 + 16) * 2 * 2
    spec = from_gguf(info)
    assert spec.layers == 6
    assert spec.full_attn_layers == 2
    assert spec.tied_embeddings
    assert any("MTP" in n for n in spec.notes)
    mtp_bytes = sum(t.nbytes for t in info.tensors if t.name.startswith("blk.6."))
    assert spec.read_bytes == info.weight_bytes - mtp_bytes


def test_moe_reads_only_used_experts(moe_gguf):
    info = read_gguf(str(moe_gguf))
    exps = sum(t.nbytes for t in info.tensors if "_exps." in t.name)
    embd = next(t for t in info.tensors if t.name == "token_embd.weight").nbytes
    assert info.bytes_read_per_token() == int(info.weight_bytes - embd - exps + exps * 2 / 8)
    spec = from_gguf(info)
    assert (spec.experts_total, spec.experts_used) == (8, 2)
    assert spec.expert_union(1) == pytest.approx(0.25)
    assert spec.expert_union(4) == pytest.approx(1 - 0.75 ** 4)
    assert spec.read_bytes_per_step("Q4_K_M", 4) > spec.read_bytes_per_step("Q4_K_M", 1)


def test_load_model_file_detects_gguf(dense_gguf):
    spec = load_model_file(str(dense_gguf))
    assert spec.file_quant == "Q4_K_M"
    assert spec.weight_bytes("Q4_K_M") == spec.file_bytes
    assert spec.kv_heads == 1 and spec.attn_heads == 4


def test_not_a_gguf(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(GGUFError):
        read_gguf(str(p))
