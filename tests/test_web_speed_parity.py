"""The web page's speed model (web/src/speed.ts) must match the Python one.

Same approach as test_upstream_parity: run the TypeScript under node's type
stripping and compare. Skipped when node >= 22.6 is missing or the web/
folder is not around (sdist installs).
"""

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from amagi_inference_calc import quant as Q
from amagi_inference_calc import speed as S
from amagi_inference_calc.cli import main
from amagi_inference_calc.data import load_json
from amagi_inference_calc.models import builtin_models

TS = Path(__file__).resolve().parents[1] / "web" / "src" / "speed.ts"
NODE = shutil.which("node")
GPUS = ["rtx-3090", "rtx-3060-12gb", "rtx-4090", "tesla-p40", "rx-7900-xtx", "apple-m4-max-128gb"]
NUMS = ["weights_gib", "kv_gib", "vram_needed_gib", "vram_total_gib", "bandwidth_gbs", "tok_s_single",
        "tok_s_per_user", "tok_s_total", "prefill_tok_s", "ttft_s", "ttft_all_s", "price_usd", "usd_per_tok_s"]


def _node_ok():
    if not NODE or not TS.exists():
        return False
    out = subprocess.run([NODE, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v")
    major, minor = (int(x) for x in out.split(".")[:2])
    return (major, minor) >= (22, 6)


def _run_ts(tmp_path, body):
    data = {"models": load_json("models.json")["models"], "gpus": load_json("gpus.json")["gpus"],
            "cal": load_json("calibration.json"), "presets": load_json("office_presets.json")["presets"]}
    (tmp_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "run.ts").write_text(
        "import { readFileSync } from 'node:fs';\n"
        f"import * as S from {json.dumps(TS.as_uri())};\n"
        "const D = JSON.parse(readFileSync(new URL('./data.json', import.meta.url), 'utf8'));\n"
        "const model = (id: string) => D.models.find((m: any) => m.id === id);\n"
        "const gpu = (id: string) => D.gpus.find((g: any) => g.id === id);\n" + body, encoding="utf-8")
    run = subprocess.run([NODE, "--experimental-strip-types", "--no-warnings", str(tmp_path / "run.ts")],
                         capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@pytest.mark.skipif(not _node_ok(), reason="needs node >= 22.6 and the web/ sources")
def test_quant_tables_match():
    src = TS.read_text(encoding="utf-8")
    for name, (bpw, _) in Q.QUANTS.items():
        assert f"  {name}: {bpw}" in src or f" {name}: {bpw}," in src, name
    for name, size in Q.KV_TYPES.items():
        assert name in src, name


@pytest.mark.skipif(not _node_ok(), reason="needs node >= 22.6 and the web/ sources")
def test_estimate_matches_python(tmp_path):
    models = [m.id for m in builtin_models()]
    cases = [dict(model=m, gpu=g, users=u, context=c, count=n, quant=q, kv=kv, price=pr)
             for m, g, u, c, n, q, kv, pr in itertools.product(
                 models, GPUS, [1, 3, 9], [4096, 32768], [1, 2], [None, "Q8_0"], ["F16", "Q8_0"], [None, 500])
             if not (q and kv == "Q8_0")]
    got = _run_ts(tmp_path, f"const cases = {json.dumps(cases)};\n"
                  "console.log(JSON.stringify(cases.map((c: any) => S.estimate(model(c.model), gpu(c.gpu), D.cal, "
                  "{users: c.users, context: c.context, gpuCount: c.count, quant: c.quant ?? undefined, "
                  "kvType: c.kv, price: c.price}))));\n")
    assert len(got) == len(cases) > 2000
    cal = S.load_calibration()
    from amagi_inference_calc.models import find_model
    for c, ts in zip(cases, got):
        py = S.estimate(find_model(c["model"]), S.find_gpu(c["gpu"]), quant=c["quant"], users=c["users"],
                        context=c["context"], gpu_count=c["count"], kv_type=c["kv"], price=c["price"], cal=cal)
        for k in NUMS:
            a, b = getattr(py, k), ts[k]
            assert (a is None) == (b is None), (c, k)
            if a is not None:
                assert b == pytest.approx(a, rel=1e-9), (c, k)
        for k in ("quant", "fits", "cards_needed", "bandwidth_source", "warnings", "prompt"):
            assert ts[k] == getattr(py, k), (c, k)


@pytest.mark.skipif(not _node_ok(), reason="needs node >= 22.6 and the web/ sources")
def test_office_picks_match_cli(tmp_path, capsys):
    presets = load_json("office_presets.json")["presets"]
    got = _run_ts(tmp_path, "const ms = S.OFFICE_MODELS.map(model);\n"
                  "console.log(JSON.stringify(D.presets.map((p: any) => S.officePicks(p, ms, D.gpus, D.cal))));\n")
    for p, ts in zip(presets, got):
        assert main(["office", p["id"], "--json"]) == 0
        py = json.loads(capsys.readouterr().out)["options"]
        assert sorted(py) == sorted(ts), p["id"]
        for mid in py:
            assert [(r["gpu"], r["gpu_count"]) for r in py[mid]] == [(r["gpu"], r["gpu_count"]) for r in ts[mid]]
            assert [round(r["tok_s_per_user"], 3) for r in py[mid]] == [round(r["tok_s_per_user"], 3) for r in ts[mid]]
