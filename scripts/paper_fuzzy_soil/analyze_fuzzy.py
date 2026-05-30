#!/usr/bin/env python3
"""
Validate the fuzzy difficulty index as a descriptor (Set A).

Reads the mu-sweep (fuzzy_sweep.csv), computes the paper's index d (eq.2) per
condition (rigid contact), and tests whether d faithfully describes measured
difficulty:

  1. Monotonicity: Spearman corr of each metric vs d. Higher d = harder, so
     slip/falls/vel_err should RISE with d, ep_len should FALL.
  2. Interpretable regimes: label each condition by its dominant fuzzy
     consequent (Easy/Moderate/Hard) and show mean performance separates across
     the three bins in the correct order — the core "fuzzy descriptor works" claim.

Scope note: under PhysX rigid contact the support axis is pinned Hard and is
physically inert (see conclusion writeup), so d is exercised here via the
traction axis only. d is a monotone transform of mu, so this validates that the
fuzzy LABELLING tracks difficulty; it does not independently validate the
support dimension (that needs deformation physics).

Usage: python analyze_fuzzy.py --sweep logs/FuzzySoilFurrows/fuzzy_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuzzy_soil as fz  # noqa: E402

# (csv col, label, expected sign of corr vs d : harder(higher d) -> ?)
METRICS = [
    ("slip_per_100m", "slip / 100m", "up"),
    ("falls_per_100m", "falls / 100m", "up"),
    ("ep_len_steps", "ep_len (steps)", "down"),
    ("vel_err", "vel error (m/s)", "up"),
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys))


def _regime(mu, kn):
    """Dominant fuzzy consequent label for (mu, kn)."""
    tl, tm, th = fz.trac_low(mu), fz.trac_med(mu), fz.trac_high(mu)
    ss, sm, sh = fz.sup_soft(kn), fz.sup_med(kn), fz.sup_hard(kn)
    rules = [
        (min(tl, ss), "Hard"), (min(tl, sm), "Hard"), (min(tl, sh), "Hard"),
        (min(tm, ss), "Moderate"), (min(tm, sm), "Moderate"), (min(tm, sh), "Moderate"),
        (min(th, ss), "Moderate"), (min(th, sm), "Easy"), (min(th, sh), "Easy"),
    ]
    g = {"Easy": 0.0, "Moderate": 0.0, "Hard": 0.0}
    for a, c in rules:
        g[c] = max(g[c], a)
    return max(g, key=g.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.sweep)) if _f(r.get("mu")) is not None]
    rows.sort(key=lambda r: _f(r["mu"]))
    if len(rows) < 3:
        print(f"[fuzzy] need >=3 conditions, got {len(rows)} — sweep incomplete?")
        return 2

    mus = [_f(r["mu"]) for r in rows]
    kn = [_f(r.get("kn_knm")) or fz.RIGID_KN_KNM for r in rows]
    d = [fz.difficulty_index(m, k) for m, k in zip(mus, kn)]
    regime = [_regime(m, k) for m, k in zip(mus, kn)]

    print("== Fuzzy difficulty index validation (Set A) — mu sweep, rigid contact ==")
    print("   Convention: higher d = HARDER soil.\n")
    hdr = ["mu", "d", "regime"] + [m[0] for m in METRICS]
    print("   " + " | ".join(f"{h:>14}" for h in hdr))
    for i, r in enumerate(rows):
        cells = [f"{mus[i]:.2f}", f"{d[i]:.3f}", regime[i]] + \
                [(f"{_f(r.get(m[0])):.2f}" if _f(r.get(m[0])) is not None else "n/a") for m in METRICS]
        print("   " + " | ".join(f"{c:>14}" for c in cells))

    print("\n   == Monotonicity vs d (Spearman; higher d = harder) ==")
    print(f"   {'metric':>16} | {'Spearman rho':>12} | dir ok?")
    n_ok = 0
    for col, label, sign in METRICS:
        ys = [_f(r.get(col)) for r in rows]
        if any(v is None for v in ys):
            print(f"   {label:>16} |   (missing data)")
            continue
        rho = _spearman(d, ys)
        ok = (rho > 0) if sign == "up" else (rho < 0)
        n_ok += int(ok and abs(rho) >= 0.6)
        print(f"   {label:>16} | {rho:>12.3f} | {'YES' if ok else 'no':>3} (want {sign} with d)")

    # Baseline: does fuzzy d add anything beyond raw mu? (Codex's circularity check.)
    # Since kn is inert in PhysX, d is a monotone transform of mu over this sweep, so
    # we expect |rho(d)| ~= |rho(mu)|. Reporting both makes the (lack of) added value explicit.
    print("\n   == Fuzzy d vs raw mu (added-value / circularity check) ==")
    print("   kn is physically inert here, so d is ~a monotone transform of mu;")
    print("   similar |rho| for both means d adds ORDERING, not new information.")
    print(f"   {'metric':>16} | {'|rho| vs d':>10} | {'|rho| vs mu':>11} | note")
    for col, label, sign in METRICS:
        ys = [_f(r.get(col)) for r in rows]
        if any(v is None for v in ys):
            print(f"   {label:>16} |   (missing data)")
            continue
        rd = abs(_spearman(d, ys))
        rmu = abs(_spearman(mus, ys))
        note = "d~=mu (no added info)" if abs(rd - rmu) < 0.1 else \
               ("d stronger" if rd > rmu else "mu stronger")
        print(f"   {label:>16} | {rd:>10.3f} | {rmu:>11.3f} | {note}")

    # Interpretable-regime separation
    print("\n   == Interpretable regime separation (mean performance per fuzzy label) ==")
    order = ["Easy", "Moderate", "Hard"]
    present = [g for g in order if g in regime]
    print(f"   {'regime':>10} | {'n':>2} | " + " | ".join(f"{m[1]:>14}" for m in METRICS))
    bin_means = {}
    for g in present:
        idx = [i for i, rg in enumerate(regime) if rg == g]
        means = []
        for col, _, _ in METRICS:
            vals = [_f(rows[i].get(col)) for i in idx if _f(rows[i].get(col)) is not None]
            means.append(sum(vals) / len(vals) if vals else float("nan"))
        bin_means[g] = means
        print(f"   {g:>10} | {len(idx):>2} | " + " | ".join(f"{m:>14.2f}" for m in means))
    # Check Easy->Hard ordering for slip (should rise) and ep_len (should fall)
    sep_ok = ""
    if {"Easy", "Hard"} <= set(bin_means):
        slip_e, slip_h = bin_means["Easy"][0], bin_means["Hard"][0]
        epl_e, epl_h = bin_means["Easy"][2], bin_means["Hard"][2]
        sep_ok = (f"   Easy->Hard: slip {slip_e:.0f}->{slip_h:.0f} "
                  f"({'rises OK' if slip_h > slip_e else 'WRONG'}), "
                  f"ep_len {epl_e:.0f}->{epl_h:.0f} "
                  f"({'falls OK' if epl_h < epl_e else 'WRONG'})")
        print(sep_ok)

    print(f"\n   VERDICT: {n_ok}/{len(METRICS)} metrics monotone in d (|rho|>=0.6, correct sign).")
    print("   HONEST SCOPE (per Codex): this validates d as an ORDINAL logging descriptor")
    print("   for THIS friction sweep only. Because kn is physically inert in PhysX, d is")
    print("   ~a monotone transform of mu, so a positive result mainly restates 'lower")
    print("   friction = harder'. It does NOT validate the 2D fuzzy model, the stiffness")
    print("   axis, or added value over raw mu (see the d-vs-mu table above).")
    if n_ok >= 3:
        print("   => d tracks measured difficulty across the friction sweep (limited claim).")
    else:
        print("   => d did NOT track difficulty cleanly even on friction; inspect the sweep.")

    out_png = args.out_png or os.path.join(os.path.dirname(args.sweep), "fuzzy_validation.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"Easy": "tab:green", "Moderate": "tab:orange", "Hard": "tab:red"}
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.8))
        for ax, (col, label, sign) in zip(axes, METRICS):
            ys = [_f(r.get(col)) for r in rows]
            if any(v is None for v in ys):
                ax.set_title(f"{label}\n(missing)"); continue
            ax.scatter(d, ys, c=[colors.get(g, "gray") for g in regime], s=70, zorder=3)
            for x, y, m in zip(d, ys, mus):
                ax.annotate(f"μ={m:.2f}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("fuzzy d (higher = harder)")
            ax.set_ylabel(label)
            ax.set_title(f"{label}\nSpearman ρ={_spearman(d, ys):.2f}")
            ax.grid(alpha=0.3)
        handles = [plt.Line2D([0], [0], marker='o', ls='', color=c, label=g) for g, c in colors.items()]
        fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8)
        fig.suptitle("Fuzzy difficulty index validation — traction sweep (rigid contact)", y=1.04)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight", dpi=130)
        print(f"\n   figure -> {out_png}")
    except Exception as e:
        print(f"\n   (figure skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
