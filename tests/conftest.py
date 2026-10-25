import pytest

from amagi_inference_calc.gguf import write_gguf

F16, Q4_K, Q6_K = 1, 12, 14


def _t(name, dims, type_id=Q4_K, nbytes=None):
    n = 1
    for d in dims:
        n *= d
    return {"name": name, "dims": dims, "type": type_id, "nbytes": nbytes if nbytes is not None else n // 2}


@pytest.fixture
def dense_gguf(tmp_path):
    """4-layer llama-style model with an untied output head."""
    emb, ff, vocab = 64, 128, 100
    tensors = [_t("token_embd.weight", [emb, vocab]), _t("output.weight", [emb, vocab], Q6_K)]
    for i in range(4):
        tensors += [
            _t(f"blk.{i}.attn_q.weight", [emb, emb]),
            _t(f"blk.{i}.attn_k.weight", [emb, 16]),
            _t(f"blk.{i}.attn_v.weight", [emb, 16]),
            _t(f"blk.{i}.attn_output.weight", [emb, emb]),
            _t(f"blk.{i}.ffn_up.weight", [emb, ff]),
            _t(f"blk.{i}.ffn_gate.weight", [emb, ff]),
            _t(f"blk.{i}.ffn_down.weight", [ff, emb]),
        ]
    meta = {
        "general.architecture": "llama", "general.name": "Tiny Dense", "general.file_type": 15,
        "llama.block_count": 4, "llama.embedding_length": emb, "llama.attention.head_count": 4,
        "llama.attention.head_count_kv": 1, "llama.context_length": 4096,
    }
    p = tmp_path / "tiny-dense-Q4_K_M.gguf"
    write_gguf(str(p), meta, tensors)
    return p


@pytest.fixture
def hybrid_gguf(tmp_path):
    """6 blocks: every third one is attention, the rest linear (ssm), plus one MTP block."""
    emb = 64
    tensors = [_t("token_embd.weight", [emb, 100])]  # tied: no output.weight
    for i in range(6):
        if i % 3 == 2:
            tensors += [_t(f"blk.{i}.attn_q.weight", [emb, emb]), _t(f"blk.{i}.attn_k.weight", [emb, 32]),
                        _t(f"blk.{i}.attn_v.weight", [emb, 32])]
        else:
            tensors += [_t(f"blk.{i}.attn_qkv.weight", [emb, 3 * emb]), _t(f"blk.{i}.ssm_a", [16], 0, 64)]
        tensors += [_t(f"blk.{i}.ffn_up.weight", [emb, 128])]
    tensors += [_t("blk.6.attn_k.weight", [emb, 32]), _t("blk.6.nextn.eh_proj.weight", [2 * emb, emb])]
    meta = {
        "general.architecture": "qwen35", "general.name": "Tiny Hybrid", "general.file_type": 15,
        "qwen35.block_count": 7, "qwen35.embedding_length": emb, "qwen35.attention.head_count": 4,
        "qwen35.attention.head_count_kv": 2, "qwen35.attention.key_length": 16,
        "qwen35.attention.value_length": 16, "qwen35.context_length": 8192,
    }
    p = tmp_path / "tiny-hybrid-Q4_K_M.gguf"
    write_gguf(str(p), meta, tensors)
    return p


@pytest.fixture
def moe_gguf(tmp_path):
    """2-layer MoE with 8 experts, 2 used per token."""
    emb, ff, n_exp = 64, 32, 8
    tensors = [_t("token_embd.weight", [emb, 100]), _t("output.weight", [emb, 100])]
    for i in range(2):
        tensors += [
            _t(f"blk.{i}.attn_q.weight", [emb, emb]), _t(f"blk.{i}.attn_k.weight", [emb, 16]),
            _t(f"blk.{i}.attn_v.weight", [emb, 16]), _t(f"blk.{i}.ffn_gate_inp.weight", [emb, n_exp], 0, emb * n_exp * 4),
            _t(f"blk.{i}.ffn_up_exps.weight", [emb, ff, n_exp]), _t(f"blk.{i}.ffn_down_exps.weight", [ff, emb, n_exp]),
        ]
    meta = {
        "general.architecture": "qwen3moe", "general.name": "Tiny MoE", "general.file_type": 15,
        "qwen3moe.block_count": 2, "qwen3moe.embedding_length": emb, "qwen3moe.attention.head_count": 4,
        "qwen3moe.attention.head_count_kv": 1, "qwen3moe.expert_count": n_exp, "qwen3moe.expert_used_count": 2,
        "qwen3moe.context_length": 32768,
    }
    p = tmp_path / "tiny-moe-Q4_K_M.gguf"
    write_gguf(str(p), meta, tensors)
    return p
