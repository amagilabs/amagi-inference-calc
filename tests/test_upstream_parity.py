"""`amagi-inference-calc vram` must match the upstream web app's math.

Runs web/src/calculations.ts (from llm-inference-calculator by Alex
Ziskind, MIT) under node's type stripping and compares every case.
Skipped unless node >= 22.6 is on PATH or the web/ folder is not around (sdist installs).
"""

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from amagi_inference_calc import upstream as U

TS = Path(__file__).resolve().parents[1] / "web" / "src" / "calculations.ts"
NODE = shutil.which("node")

CASES = [
    dict(params=p, quant=q, ctx=c, kv=kv, kvq=kq, mem=mm, sysmem=sm, vram=v, mode=mo)
    for p, q, c, kv, kq, mm, sm, v, mo in itertools.product(
        [1.5, 7, 70], ["F16", "Q8", "Q4", "Q2", "GPTQ", "AWQ"], [2048, 32768], [True, False],
        ["F16", "Q4"], ["DISCRETE_GPU", "UNIFIED_MEMORY"], [64], [12, 24], ["incremental", "bulk"])
]


def _node_ok():
    if not NODE or not TS.exists():
        return False
    out = subprocess.run([NODE, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v")
    major, minor = (int(x) for x in out.split(".")[:2])
    return (major, minor) >= (22, 6)


@pytest.mark.skipif(not _node_ok(), reason="needs node >= 22.6 and the web/ sources")
def test_matches_upstream_typescript(tmp_path):
    src = "\n".join(ln for ln in TS.read_text(encoding="utf-8").splitlines()
                    if not ln.strip().startswith(("import ", "} from", "MemoryMode,", "ModelQuantization,",
                                                  "KvCacheQuantization,", "Recommendation,")))
    src = src.replace("): Recommendation", "): any")
    (tmp_path / "calc.ts").write_text(src, encoding="utf-8")
    (tmp_path / "cases.json").write_text(json.dumps(CASES), encoding="utf-8")
    (tmp_path / "run.ts").write_text(
        "import { readFileSync } from 'node:fs';\n"
        "import { calculateHardwareRecommendation, calculateOnDiskSize } from './calc.ts';\n"
        "const cases = JSON.parse(readFileSync(new URL('./cases.json', import.meta.url), 'utf8'));\n"
        "console.log(JSON.stringify(cases.map((c: any) => ({rec: calculateHardwareRecommendation(c.params, c.quant,"
        " c.ctx, c.kv, c.kvq, c.mem, c.sysmem, c.vram, c.mode), disk: calculateOnDiskSize(c.params, c.quant)}))));\n",
        encoding="utf-8")
    run = subprocess.run([NODE, "--experimental-strip-types", "--no-warnings", str(tmp_path / "run.ts")],
                         capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    expected = json.loads(run.stdout)
    assert len(expected) == len(CASES)
    for c, exp in zip(CASES, expected):
        got = U.hardware_recommendation(c["params"], c["quant"], c["ctx"], c["kv"], c["kvq"],
                                        c["mem"], c["sysmem"], c["vram"], c["mode"])
        assert got == exp["rec"], c
        assert U.on_disk_size(c["params"], c["quant"]) == pytest.approx(exp["disk"]), c


def test_known_values():
    # 7B at Q4 with an f16 KV cache, 4K context: (3.5 + 3.5*0.2*2*2) * 1.1
    assert U.required_vram(7, "Q4", 4096, True, "F16") == pytest.approx((3.5 + 3.5 * 0.2 * 2 * 2) * 1.1)
    # 70B: 100.1 GB needed, upstream adds 20% headroom per split: ceil(100.1*1.2/24) = 6
    rec = U.hardware_recommendation(70, "Q4", 8192, True, "F16", "DISCRETE_GPU", 64, 24)
    assert rec["vramNeeded"] == "100.10"
    assert rec["gpuType"] == "6x 24GB GPUs" and rec["gpusRequired"] == 6
