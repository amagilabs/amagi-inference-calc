import pytest

from amagi_inference_calc import speed as S
from amagi_inference_calc.data import load_json
from amagi_inference_calc.models import find_model


def test_shipped_calibration_is_the_fit():
    """calibration.json must be exactly what fit() gives on measurements.json."""
    fitted, shipped = S.fit(), S.load_calibration()
    for k in ("bw_eff_gbs", "overhead_ms", "moe_layer_ms", "batch_ms_per_b_dense",
              "batch_ms_per_layer_moe", "tflops_dense", "tflops_moe"):
        assert getattr(fitted, k) == pytest.approx(getattr(shipped, k), abs=1e-3), k


def test_dense_fit_is_close_to_memtest_bandwidth():
    cal = S.load_calibration()
    # decode reaches about the memtest_vulkan figure, well under the spec sheet
    assert 0.95 < cal.bw_eff_gbs / cal.bw_ref_gbs < 1.10
    assert 0.75 < cal.decode_eff < 0.85


def _rows(kind, role, moe=None, multi=None):
    out = []
    for r in S.backtest(role=role):
        if r["kind"] != kind:
            continue
        m = find_model(r["model"])
        if moe is not None and bool(m.experts_total) != moe:
            continue
        if multi is not None and (" 1 stream" not in r["where"]) == (not multi):
            continue
        out.append(r)
    assert out, (kind, role, moe, multi)
    return out


def test_fit_rows_reproduce():
    for r in _rows("decode", "fit", multi=False):
        assert abs(r["error_pct"]) < 1.5, r


def test_held_out_single_user_decode_within_10pct():
    for r in _rows("decode", "backtest", multi=False):
        assert abs(r["error_pct"]) < 10, r


def test_held_out_multi_user_moe_decode_within_10pct():
    for r in _rows("decode", "backtest", multi=True):
        assert abs(r["error_pct"]) < 10, r


def test_held_out_dense_prefill_within_10pct():
    for r in _rows("prefill", "backtest", moe=False):
        assert abs(r["error_pct"]) < 10, r


def test_moe_prefill_is_rough_and_documented():
    worst = max(abs(r["error_pct"]) for r in _rows("prefill", "backtest", moe=True))
    assert worst < 70  # docs/CALCULATIONS.md says "up to about 60%"


def test_measured_vram_is_covered():
    """The VRAM estimate is never below what nvidia-smi showed, and not far above."""
    gpu = S.find_gpu("rtx-3090")
    for p in load_json("measurements.json")["vram_points"]:
        e = S.estimate(find_model(p["model"]), gpu, quant=p["quant"], context=p["context"])
        measured_gib = p["mib"] / 1024
        assert measured_gib <= e.vram_needed_gib <= measured_gib + 0.6, p


def test_estimate_basics():
    m, g = find_model("qwen3-14b"), S.find_gpu("rtx-3090")
    one = S.estimate(m, g, users=1, context=8192)
    five = S.estimate(m, g, users=5, context=8192)
    assert one.fits and five.fits
    assert five.tok_s_per_user < one.tok_s_per_user < five.tok_s_total
    assert five.kv_gib == pytest.approx(5 * one.kv_gib)
    assert one.prefill_tok_s and one.ttft_s > 0
    assert five.ttft_all_s == pytest.approx(5 * five.ttft_s)
    assert one.usd_per_tok_s == pytest.approx(g.price_used_usd / one.tok_s_total)


def test_layer_split_adds_memory_not_speed():
    m, g = find_model("llama-3.3-70b"), S.find_gpu("rtx-3090")
    one, two = S.estimate(m, g, gpu_count=1), S.estimate(m, g, gpu_count=2)
    assert not one.fits and one.cards_needed == 2 and one.usd_per_tok_s is None
    assert two.fits
    assert two.tok_s_single == pytest.approx(one.tok_s_single)
    assert two.price_usd == 2 * g.price_used_usd


def test_uncalibrated_gpu_scales_by_spec_and_skips_prefill():
    cal = S.load_calibration()
    g = S.find_gpu("rtx-4090")
    e = S.estimate(find_model("qwen3-8b"), g)
    assert e.bandwidth_gbs == pytest.approx(g.bw_spec_gbs * cal.decode_eff)
    assert e.prefill_tok_s is None
    assert S.estimate(find_model("qwen3-8b"), g, tflops=150).prefill_tok_s > 0


def test_memtest_bandwidth_beats_spec_scaling():
    g = S.find_gpu("rtx-3060-12gb")
    g.bw_measured_gbs = 300.0
    e = S.estimate(find_model("qwen3-8b"), g)
    assert "memtest" in e.bandwidth_source
    assert e.bandwidth_gbs == pytest.approx(300 * S.load_calibration().bw_eff_gbs / S.load_calibration().bw_ref_gbs)


def test_non_nvidia_and_context_warnings():
    e = S.estimate(find_model("qwen3-8b"), S.find_gpu("rx-7900-xtx"), context=65536)
    text = " ".join(e.warnings)
    assert "AMD" in text and "trained" in text


def test_find_gpu():
    assert S.find_gpu("3090").id == "rtx-3090"
    assert S.find_gpu("RTX 3090 Ti").id == "rtx-3090-ti"
    with pytest.raises(KeyError):
        S.find_gpu("rtx-3060")  # 12 GB or Ti?
    with pytest.raises(KeyError):
        S.find_gpu("voodoo-5")


def test_decode_bw_from_runs_recovers_calibration():
    cal = S.load_calibration()
    m = find_model("qwen2.5-32b")
    bw = cal.bw_eff_gbs * 1e9
    runs = [(c, 1 / S.decode_step_s(m, "Q4_K_M", c, 1, bw, cal)) for c in (200, 2000, 8000)]
    assert S.decode_bw_from_runs(m, "Q4_K_M", runs, cal) == pytest.approx(cal.bw_eff_gbs, rel=1e-6)


def test_tokbench_runs():
    rep = {"cases": [{"median": {"prompt_tokens": 100, "gen_tokens": 200, "gen_tps": 50.0}},
                     {"median": {"prompt_tokens": 600, "gen_tokens": 100, "gen_tps": 40.0, "server_gen_tps": 41.0}},
                     {"concurrency": 4, "median": {"prompt_tokens": 600, "gen_tokens": 100, "gen_tps": 20.0}}]}
    assert S.tokbench_runs(rep) == [(200, 50.0), (650, 41.0)]
