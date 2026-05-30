#!/usr/bin/env python3
"""
Arm B vs Arm C (hard-gate ablation) slip analysis.

Codex's methodological point (docs/experiments/fuzzy_soil_furrows_result.md):
the headline slip_per_100m confounds gate-effect with survival — the arm that
walks farther accumulates more tangential foot travel, so a naive per-100m
comparison can't tell "the gate worsened slip" from "this arm just walked more."

This script defuses that at the PER-EPISODE level (the granularity sample_eps.py
records). It reports, for each arm:
  - aggregate slip_per_100m (= sum slip / sum distance * 100)
  - an OLS regression slip_distance ~ distance  (slope = marginal slip per meter)
  - matched-distance bins: mean slip_per_100m within shared distance bins, so B
    and C are compared at equal distance-traveled.

Within-episode windowing (first 0.72 m of every episode) would need per-step
logging and is intentionally NOT done here — reserved for a 3-seed confirmation
if Arm C looks consequential.

Usage:
    python analyze_armC.py --b per_episode_B_seed1.csv --c per_episode_C_seed1.csv --seed 1
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            out.append({
                "distance": float(r["distance"]),
                "slip": float(r["slip_distance"]),
                "ep_len": float(r["episode_length"]),
                "fell": int(float(r["fell"])),
            })
        except (KeyError, ValueError):
            continue
    return out


def agg_slip_per_100m(eps: list[dict]) -> float:
    td = sum(e["distance"] for e in eps)
    ts = sum(e["slip"] for e in eps)
    return (ts / (td / 100.0)) if td > 0 else float("nan")


def ols(eps: list[dict]) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) for slip ~ distance."""
    xs = [e["distance"] for e in eps]
    ys = [e["slip"] for e in eps]
    n = len(xs)
    if n < 2:
        return (float("nan"),) * 3
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return (float("nan"),) * 3
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def matched_bins(b: list[dict], c: list[dict], n_bins: int = 5):
    """Compare slip_per_100m within shared distance bins."""
    all_d = [e["distance"] for e in b + c if e["distance"] > 0]
    if not all_d:
        return []
    lo, hi = min(all_d), max(all_d)
    if hi <= lo:
        return []
    width = (hi - lo) / n_bins
    rows = []
    for i in range(n_bins):
        b_lo = lo + i * width
        b_hi = lo + (i + 1) * width if i < n_bins - 1 else hi + 1e-9
        bb = [e for e in b if b_lo <= e["distance"] < b_hi]
        cc = [e for e in c if b_lo <= e["distance"] < b_hi]
        rows.append({
            "range": f"[{b_lo:.2f},{b_hi:.2f})",
            "n_b": len(bb), "n_c": len(cc),
            "slip100_b": agg_slip_per_100m(bb) if bb else float("nan"),
            "slip100_c": agg_slip_per_100m(cc) if cc else float("nan"),
        })
    return rows


def fmt(x: float) -> str:
    return "n/a" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", required=True, help="Arm B per_episode.csv")
    ap.add_argument("--c", required=True, help="Arm C per_episode.csv")
    ap.add_argument("--seed", default="?")
    args = ap.parse_args()

    b, c = load(args.b), load(args.c)
    print(f"== Arm B vs Arm C (hard-gate ablation) — seed {args.seed} ==")
    print(f"   Arm B: {len(b)} episodes loaded from {args.b}")
    print(f"   Arm C: {len(c)} episodes loaded from {args.c}")
    if not c:
        print("   ERROR: Arm C per-episode data missing — cannot analyze.")
        return 2

    def block(name, eps):
        if not eps:
            print(f"\n   [{name}] no data")
            return
        td = sum(e["distance"] for e in eps)
        avg_d = td / len(eps)
        avg_len = sum(e["ep_len"] for e in eps) / len(eps)
        slope, intercept, r2 = ols(eps)
        print(f"\n   [{name}]")
        print(f"     avg ep_len      : {avg_len:.1f} steps")
        print(f"     avg distance    : {avg_d:.3f} m")
        print(f"     slip_per_100m   : {fmt(agg_slip_per_100m(eps))}")
        print(f"     slip~dist slope : {fmt(slope)} m-slip per m-traveled  (r2={fmt(r2)})")
        print(f"     slip~dist icept : {fmt(intercept)}")

    block("Arm B (soft gate)", b)
    block("Arm C (hard gate)", c)

    print("\n   == Matched-distance bins (slip_per_100m within equal distance ranges) ==")
    print(f"     {'range':>16} | {'n_B':>5} {'n_C':>5} | {'slip100_B':>10} {'slip100_C':>10}")
    for r in matched_bins(b, c):
        print(f"     {r['range']:>16} | {r['n_b']:>5} {r['n_c']:>5} | "
              f"{fmt(r['slip100_b']):>10} {fmt(r['slip100_c']):>10}")

    # Directional read
    print("\n   == Directional read ==")
    sl_b, sl_c = agg_slip_per_100m(b), agg_slip_per_100m(c)
    slope_b = ols(b)[0]
    slope_c = ols(c)[0]
    if b and not math.isnan(sl_b) and not math.isnan(sl_c):
        print(f"     aggregate slip_per_100m:  B={fmt(sl_b)}  C={fmt(sl_c)}  "
              f"(C/B = {fmt(sl_c/sl_b) if sl_b else 'n/a'})")
    if not math.isnan(slope_b) and not math.isnan(slope_c):
        print(f"     marginal slip/meter    :  B={fmt(slope_b)}  C={fmt(slope_c)}  "
              f"(C/B = {fmt(slope_c/slope_b) if slope_b else 'n/a'})")
        print("     NOTE: the slope is the confound-controlled metric — it measures slip")
        print("           per additional meter, independent of how far each arm walked.")
        if slope_c < slope_b:
            print("     => Hard gate lowers marginal slip/m: the SOFT gate was contributing slip.")
        else:
            print("     => Hard gate does NOT lower marginal slip/m: gate is not the slip driver;")
            print("        the rigid-soil / missing-compliance hypothesis moves to the front.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
