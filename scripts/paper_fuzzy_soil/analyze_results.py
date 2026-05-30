#!/usr/bin/env python3
"""
Analyze the fuzzy-soil-on-furrows A/B results and report the preregistered
decision rule.

Preregistered rule (docs/experiments/fuzzy_soil_furrows_plan.md):
    Arm B "works" iff
        (mean_ep_len_B  >=  mean_ep_len_A  - 1*SE_A)
      AND
        (slip_per_100m_B  <=  0.5 * slip_per_100m_A)

Reads logs/FuzzySoilFurrows/results.csv and prints:
    - per-arm mean ± SE of every metric
    - per-seed table for outlier inspection
    - pass / fail / inconclusive verdict on the rule

Usage:
    python scripts/paper_fuzzy_soil/analyze_results.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from statistics import mean

RESULTS = Path("logs/FuzzySoilFurrows/results.csv")


def _safe_float(x: str) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var) / math.sqrt(len(values))


def main() -> int:
    if not RESULTS.exists():
        print(f"[analyze] missing {RESULTS} — eval has not run yet")
        return 2

    with RESULTS.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[analyze] {RESULTS} is empty")
        return 2

    by_arm: dict[str, list[dict]] = {"A": [], "B": []}
    for r in rows:
        arm = r.get("arm", "").strip()
        if arm in by_arm:
            by_arm[arm].append(r)

    metrics = ("ep_len_steps", "distance_m", "slip_per_100m", "falls_per_100m", "falls")
    print(f"== Per-seed table ({RESULTS}) ==")
    hdr = ["arm", "seed", *metrics]
    print(" | ".join(f"{h:>16}" for h in hdr))
    for arm in ("A", "B"):
        for r in by_arm[arm]:
            cells = [arm, r.get("seed", "?")] + [r.get(m, "") for m in metrics]
            print(" | ".join(f"{c:>16}" for c in cells))

    print()
    print("== Per-arm mean ± SE ==")
    print(" | ".join(f"{h:>16}" for h in ["arm", "n", *metrics]))
    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for arm in ("A", "B"):
        rs = by_arm[arm]
        stats[arm] = {}
        n_seeds = len(rs)
        cells = [arm, str(n_seeds)]
        for m in metrics:
            vals = [_safe_float(r.get(m, "")) for r in rs]
            vals = [v for v in vals if v is not None]
            if not vals:
                stats[arm][m] = (float("nan"), float("nan"))
                cells.append("n/a")
                continue
            mu = mean(vals)
            se = _sem(vals)
            stats[arm][m] = (mu, se)
            cells.append(f"{mu:.3f}±{se:.3f}" if not math.isnan(se) else f"{mu:.3f}")
        print(" | ".join(f"{c:>16}" for c in cells))

    print()
    print("== Preregistered decision rule ==")
    ep_a, se_a = stats["A"].get("ep_len_steps", (float("nan"),)*2)
    ep_b, _   = stats["B"].get("ep_len_steps", (float("nan"),)*2)
    sl_a, _   = stats["A"].get("slip_per_100m", (float("nan"),)*2)
    sl_b, _   = stats["B"].get("slip_per_100m", (float("nan"),)*2)

    rule1_ok = (not math.isnan(ep_a) and not math.isnan(ep_b)
                and not math.isnan(se_a)
                and ep_b >= ep_a - se_a)
    rule2_ok = (not math.isnan(sl_a) and not math.isnan(sl_b)
                and sl_a > 0
                and sl_b <= 0.5 * sl_a)

    print(f"  Rule 1: ep_len_B ({ep_b:.1f}) >= ep_len_A ({ep_a:.1f}) - SE_A ({se_a:.1f})")
    print(f"          → threshold {ep_a - se_a:.1f}    →  {'PASS' if rule1_ok else 'FAIL'}")
    print(f"  Rule 2: slip_B ({sl_b:.2f}) <= 0.5 * slip_A ({sl_a:.2f})")
    print(f"          → threshold {0.5 * sl_a:.2f}    →  {'PASS' if rule2_ok else 'FAIL'}")
    print()

    if math.isnan(ep_a) or math.isnan(ep_b) or math.isnan(sl_a) or math.isnan(sl_b):
        verdict = "INCONCLUSIVE — missing metric(s); cannot evaluate rule"
    elif rule1_ok and rule2_ok:
        verdict = "PASS — Arm B (paper recipe) transfers to furrows"
    elif rule2_ok and not rule1_ok:
        verdict = "PARTIAL — slip target met, but ep_len target missed (curriculum hurts tracking)"
    elif rule1_ok and not rule2_ok:
        verdict = "PARTIAL — ep_len target met, but slip target missed (gating didn't help)"
    else:
        verdict = "FAIL — recipe does not transfer cleanly on this terrain"

    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
