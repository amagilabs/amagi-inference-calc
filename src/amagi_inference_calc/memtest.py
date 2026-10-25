"""Read measured memory bandwidth out of a memtest_vulkan log.

memtest_vulkan (github.com/GpuZelenograd/memtest_vulkan) prints one line
per pass with a "checked: <GB> <GB/s>" read figure. The first pass is a
tiny warm-up, so it is dropped and the median of the rest is used.
Terminal escape codes from `script` recordings are stripped first.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import List

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\[[0-9;?]*[A-Za-z]")
_CHECKED = re.compile(r"checked:\s*[\d.]+\s*GB\s+([\d.]+)\s*GB/s")
_DEVICE = re.compile(r"test of \d+:.*?\s(\d+)GB\s+(.+?)\s*$", re.M)


@dataclass
class MemtestResult:
    device: str
    vram_gb: int
    passes: int
    median_gbs: float
    min_gbs: float
    max_gbs: float
    errors: bool
    samples: List[float] = field(default_factory=list)

    def to_dict(self):
        d = dict(self.__dict__)
        d.pop("samples")
        return d


class MemtestError(ValueError):
    pass


def parse_memtest(text: str) -> MemtestResult:
    clean = _ANSI.sub("", text)
    vals = [float(v) for v in _CHECKED.findall(clean)]
    if not vals:
        raise MemtestError("no 'checked: ... GB/s' lines found; is this a memtest_vulkan log?")
    used = vals[1:] if len(vals) > 1 else vals
    dev = _DEVICE.search(clean)
    errors = any("error" in ln.lower() and "no errors" not in ln.lower() and "no any errors" not in ln.lower()
                 for ln in clean.splitlines())
    return MemtestResult(
        device=dev.group(2).strip() if dev else "unknown",
        vram_gb=int(dev.group(1)) if dev else 0,
        passes=len(vals),
        median_gbs=round(statistics.median(used), 1),
        min_gbs=min(used),
        max_gbs=max(used),
        errors=errors,
        samples=vals,
    )


def read_memtest(path: str) -> MemtestResult:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse_memtest(fh.read())
