from pathlib import Path

import pytest

from amagi_inference_calc.memtest import MemtestError, parse_memtest, read_memtest
from amagi_inference_calc.price import parse_overrides

FIX = Path(__file__).parent / "fixtures"


def test_real_memtest_log():
    r = read_memtest(str(FIX / "memtest_vulkan_rtx3090.log"))
    assert r.device == "NVIDIA GeForce RTX 3090" and r.vram_gb == 24
    assert r.passes == 30
    assert r.median_gbs == 739.3  # warm-up pass dropped
    assert not r.errors


def test_memtest_errors_and_garbage():
    txt = ("Standard 5-minute test of 1: Bus=0x01:00 DevId=0x2204   24GB NVIDIA GeForce RTX 3090\n"
           "  NO ERRORS  written: 18.8GB 650.2GB/s  checked: 22.5GB 711.3GB/s\n"
           "  ERROR 3 bits flipped  written: 18.8GB 650.2GB/s  checked: 22.5GB 600.0GB/s\n"
           "  NO ERRORS  written: 18.8GB 650.2GB/s  checked: 22.5GB 610.0GB/s\n")
    r = parse_memtest(txt)
    assert r.errors and r.median_gbs == 605.0
    with pytest.raises(MemtestError):
        parse_memtest("hello")


def test_price_overrides(tmp_path):
    csvf = tmp_path / "p.csv"
    csvf.write_text("gpu,usd\n# mine\nrtx-3090,\"$1,050\"\nrtx-4090,2900\n")
    out = parse_overrides(["rtx-3090=999", "p40=$300"], str(csvf))
    assert out == {"rtx-3090": 999.0, "rtx-4090": 2900.0, "p40": 300.0}
    with pytest.raises(ValueError):
        parse_overrides(["rtx-3090"])
