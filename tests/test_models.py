import json

import pytest

from amagi_inference_calc.models import builtin_models, find_model, from_hf_config, load_model_file

# bf16 safetensors "total_size" from each repo's model.safetensors.index.json
# (fetched 2026-10-25); bytes / 2 = parameter count.
BF16_TOTAL_SIZE = {
    "qwen3-8b": 16381470720,
    "qwen3-14b": 29536614400,
    "qwen3-32b": 65524246528,
    "qwen2.5-14b": 29540067328,
    "qwen2.5-72b": 145412407296,
    "qwen3-30b-a3b": 61064245248,
    "llama-3.1-8b": 16060522496,
    "llama-3.3-70b": 141107412992,
    "mistral-small-24b": 47144806400,
    "phi-4": 29319014400,
    "gemma-3-12b": 24374650080,
    "gemma-3-27b": 54864813280,
}


@pytest.mark.parametrize("mid,total", sorted(BF16_TOTAL_SIZE.items()))
def test_param_counts_match_published_weights(mid, total):
    m = find_model(mid)
    published = total / 2
    # gemma's published size includes the vision tower, which we do not count
    tol = 0.05 if mid.startswith("gemma") else 0.005
    assert m.params == pytest.approx(published, rel=tol)
    if mid.startswith("gemma"):
        assert m.params < published


def test_builtin_models_are_complete():
    ms = builtin_models()
    assert len(ms) >= 15
    for m in ms:
        assert m.params > 0 and m.layers > 0 and m.kv_heads > 0 and m.attn_heads > 0
        assert m.full_attn_layers + m.window_attn_layers <= m.layers
        assert 0 < m.active_params <= m.params


def test_find_model_errors():
    with pytest.raises(KeyError):
        find_model("no-such-model-xyz")
    with pytest.raises(KeyError, match="several"):
        find_model("qwen3")


LLAMA_LIKE = {
    "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
    "vocab_size": 128256, "intermediate_size": 14336, "tie_word_embeddings": False,
    "max_position_embeddings": 131072,
}


def test_hf_config_dense_by_hand():
    m = from_hf_config(LLAMA_LIKE, "llama-like")
    attn = 4096 * 4096 * 2 + 2 * 4096 * 8 * 128
    mlp = 3 * 4096 * 14336
    assert m.params == 32 * (attn + mlp) + 2 * 128256 * 4096
    assert m.kv_bytes(1000) == 32 * 8 * 256 * 2 * 1000
    assert m.window == 0 and m.full_attn_layers == 32


def test_hf_config_tie_default_is_true():
    cfg = dict(LLAMA_LIKE)
    cfg.pop("tie_word_embeddings")
    assert from_hf_config(cfg).tied_embeddings


def test_hf_config_sliding_window_layers():
    cfg = dict(LLAMA_LIKE, sliding_window=1024,
               layer_types=["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 26)
    m = from_hf_config(cfg)
    assert (m.full_attn_layers, m.window_attn_layers, m.window) == (1, 31, 1024)
    # KV for sliding layers stops growing at the window
    assert m.kv_bytes(100_000) - m.kv_bytes(50_000) == pytest.approx(1 * 8 * 256 * 2 * 50_000)


def test_hf_config_moe():
    cfg = {"hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4,
           "head_dim": 128, "vocab_size": 151936, "moe_intermediate_size": 768, "num_experts": 128,
           "num_experts_per_tok": 8, "tie_word_embeddings": False}
    m = from_hf_config(cfg, "moe")
    assert m.expert_params == 48 * 128 * 3 * 2048 * 768
    assert m.active_params < m.params / 5


def test_load_hf_config_file(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(LLAMA_LIKE))
    assert load_model_file(str(p)).layers == 32


def test_spec_json_roundtrip(tmp_path):
    m = find_model("qwen3-14b")
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(m.to_dict()))
    assert load_model_file(str(p)).params == m.params


def test_gpt_oss_layout():
    m = find_model("gpt-oss-20b")
    assert m.native_quant == "MXFP4"
    assert (m.experts_total, m.experts_used) == (32, 4)
    assert (m.full_attn_layers, m.window_attn_layers, m.window) == (12, 12, 128)
