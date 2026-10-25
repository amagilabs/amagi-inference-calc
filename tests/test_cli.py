import json
from pathlib import Path

import pytest

from amagi_inference_calc import __version__
from amagi_inference_calc.cli import main

FIX = Path(__file__).parent / "fixtures"


def run(capsys, *args):
    code = main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_estimate_text(capsys):
    code, out, _ = run(capsys, "estimate", "qwen3-14b", "--gpu", "rtx-3090", "--users", "5", "--context", "16384")
    assert code == 0
    assert "fits" in out and "tok/s each with 5 at once" in out and "per tok/s" in out


def test_estimate_json_and_price_override(capsys):
    code, out, _ = run(capsys, "estimate", "gpt-oss-20b", "-g", "3090", "--usd", "1000", "--json")
    d = json.loads(out)
    assert code == 0 and d["quant"] == "MXFP4" and d["price_usd"] == 1000
    assert d["usd_per_tok_s"] == pytest.approx(1000 / d["tok_s_total"], rel=1e-3)


def test_price_override_label_and_short_names(capsys, tmp_path):
    code, out, _ = run(capsys, "estimate", "qwen3-14b", "-g", "3090", "--price", "3090=900")
    assert code == 0 and "$900 (your price)" in out
    csv = tmp_path / "p.csv"
    csv.write_text("gpu,usd\nrtx-3090,800\n", encoding="utf-8")
    code, out, _ = run(capsys, "estimate", "qwen3-14b", "-g", "rtx-3090", "--prices", str(csv))
    assert code == 0 and "$800 (your price)" in out
    code, out, _ = run(capsys, "estimate", "qwen3-14b", "-g", "rtx-3090")
    assert "(used, " in out
    code, _, err = run(capsys, "estimate", "qwen3-14b", "-g", "rtx-3090", "--price", "nosuchcard=5")
    assert code != 0 and "nosuchcard" in err


def test_estimate_custom_card(capsys):
    code, out, _ = run(capsys, "estimate", "qwen3-8b", "--vram", "20", "--bw", "500", "--json")
    d = json.loads(out)
    assert code == 0 and d["gpu"] == "custom" and d["vram_total_gib"] == 20


def test_estimate_gguf_file(capsys, dense_gguf):
    code, out, _ = run(capsys, "estimate", str(dense_gguf), "-g", "rtx-3090", "--json")
    assert code == 0 and json.loads(out)["quant"] == "Q4_K_M"


def test_estimate_does_not_fit(capsys):
    code, out, _ = run(capsys, "estimate", "llama-3.3-70b", "-g", "rtx-3090")
    assert code == 0 and "does NOT fit (needs 2 card(s))" in out


def test_unknown_model_is_a_clean_error(capsys):
    code, _, err = run(capsys, "estimate", "nope-model", "-g", "rtx-3090")
    assert code == 2 and "unknown model" in err


def test_office(capsys):
    code, out, _ = run(capsys, "office", "team-5")
    assert code == 0 and "5-person office" in out and "qwen3-14b" in out and "NVIDIA only" in out
    code, out, _ = run(capsys, "office", "doc-review", "--json", "--model", "qwen3-14b", "--top", "2")
    d = json.loads(out)
    assert d["preset"]["context"] == 32768
    opts = d["options"]["qwen3-14b"]
    assert 1 <= len(opts) <= 2 and all(o["tok_s_per_user"] >= 15 for o in opts)
    assert opts == sorted(opts, key=lambda o: o["price_usd"])


def test_office_unknown_preset(capsys):
    with pytest.raises(SystemExit):
        main(["office", "castle"])


def test_listings(capsys):
    code, out, _ = run(capsys, "gpus", "--price", "rtx-3090=1000", "--json")
    d = json.loads(out)
    assert code == 0 and next(g for g in d["gpus"] if g["id"] == "rtx-3090")["price_used_usd"] == 1000
    code, out, _ = run(capsys, "models")
    assert code == 0 and "gpt-oss-20b" in out


def test_gguf_command(capsys, hybrid_gguf):
    code, out, _ = run(capsys, "gguf", str(hybrid_gguf), "--json")
    d = json.loads(out)
    assert code == 0 and d["kv_layers"] == 2 and d["mtp_layers"] == 1


def test_bandwidth_command(capsys):
    code, out, _ = run(capsys, "bandwidth", str(FIX / "memtest_vulkan_rtx3090.log"))
    assert code == 0 and "739.3 GB/s" in out and "--bw 739.3" in out


def test_calibrate_commands(capsys, tmp_path):
    code, out, _ = run(capsys, "calibrate", "--role", "backtest")
    assert code == 0 and "backtest" in out and "fit " not in out
    out_file = tmp_path / "cal.json"
    code, _, _ = run(capsys, "calibrate", "--refit", "--out", str(out_file), "--json")
    assert code == 0 and json.loads(out_file.read_text())["gpu"] == "rtx-3090"
    code, out, _ = run(capsys, "estimate", "qwen3-8b", "-g", "rtx-3090", "--calibration", str(out_file))
    assert code == 0


def test_calibrate_tokbench(capsys, tmp_path):
    rep = {"backend": {"model_file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "quant": "Q4_K_M"},
           "cases": [{"median": {"prompt_tokens": 70, "gen_tokens": 256, "gen_tps": 141.9}},
                     {"median": {"prompt_tokens": 637, "gen_tokens": 128, "gen_tps": 140.9}}]}
    p = tmp_path / "run.json"
    p.write_text(json.dumps(rep))
    code, out, _ = run(capsys, "calibrate", "--tokbench", str(p), "--json")
    d = json.loads(out)
    assert code == 0 and d["model"] == "qwen2.5-7b" and 700 < d["decode_bw_gbs"] < 800


def test_vram_upstream(capsys):
    code, out, _ = run(capsys, "vram", "--params", "7", "--json")
    d = json.loads(out)
    assert code == 0 and d["gpuType"] == "Single 24GB GPU" and d["onDiskSizeGB"] == 3.5
