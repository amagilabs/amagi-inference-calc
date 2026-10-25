"""Used-GPU prices: built-in snapshot plus user overrides.

The built-in prices are the lowest current *used* offer listed on
gpudojo.com on the date in gpus.json. They are asking prices, not sold
prices, and they move every week. Pass your own with
`--price rtx-3090=1100` or a two-column CSV via `--prices FILE`.
"""

from __future__ import annotations

import csv
from typing import Dict, Iterable, Optional

from .data import load_json


def price_date() -> str:
    return str(load_json("gpus.json").get("price_date", ""))


def parse_overrides(pairs: Iterable[str] = (), path: Optional[str] = None) -> Dict[str, float]:
    """`gpu=USD` pairs and/or a CSV of `gpu,usd` rows -> {gpu_id: usd}."""
    out: Dict[str, float] = {}
    if path:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if not row or row[0].strip().startswith("#") or len(row) < 2:
                    continue
                try:
                    out[row[0].strip()] = float(row[1].replace("$", "").replace(",", ""))
                except ValueError:
                    continue  # header row
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"price override {pair!r} must look like gpu=USD")
        k, v = pair.split("=", 1)
        out[k.strip()] = float(v.replace("$", "").replace(",", ""))
    return out
