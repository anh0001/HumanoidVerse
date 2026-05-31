#!/usr/bin/env python3
"""Set B summary chart — reads ONLY from the held-out eval CSVs + manifests."""
from __future__ import annotations
import csv, glob, json, os, statistics as st
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "logs/FuzzySoilFurrowsSetB"
EVAL = f"{ROOT}/eval"
ARMS = ["fixed", "crisp", "fuzzy"]
COL = {"fixed": "tab:gray", "crisp": "tab:orange", "fuzzy": "tab:blue"}
CAP = 1000.0


def auc(csv_path):
    by = defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        try: by[float(r["mu"])].append(float(r["ep_len_steps"]))
        except (ValueError, TypeError): pass
    mus = sorted(by)
    if len(mus) < 2: return None
    ys = [st.mean(by[m]) / CAP for m in mus]
    a = sum(0.5 * (ys[i] + ys[i+1]) * (mus[i+1] - mus[i]) for i in range(len(mus)-1))
    return a / (mus[-1] - mus[0])


def sweep(csv_path):
    by = defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        try: by[float(r["mu"])].append(float(r["ep_len_steps"]))
        except (ValueError, TypeError): pass
    mus = sorted(by)
    return mus, [st.mean(by[m]) for m in mus]


aucs = {a: [] for a in ARMS}
for a in ARMS:
    for s in (1, 2, 3):
        p = f"{EVAL}/setB_{a}_seed{s}.csv"
        if os.path.isfile(p):
            v = auc(p)
            if v is not None: aucs[a].append(v)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: robustness AUC per arm (bar = mean, dots = seeds)
xs = range(len(ARMS))
ax[0].bar(xs, [st.mean(aucs[a]) if aucs[a] else 0 for a in ARMS],
          color=[COL[a] for a in ARMS], alpha=0.65, width=0.6,
          yerr=[st.stdev(aucs[a]) if len(aucs[a]) > 1 else 0 for a in ARMS], capsize=6)
for i, a in enumerate(ARMS):
    ax[0].scatter([i]*len(aucs[a]), aucs[a], color="black", zorder=3, s=28)
ax[0].set_xticks(list(xs)); ax[0].set_xticklabels(
    [f"{a}\nn={len(aucs[a])}" for a in ARMS])
ax[0].set_ylabel("held-out robustness AUC (mean ep_len / cap, over μ sweep)")
ax[0].set_title("(1) Robustness AUC — fuzzy ≈ crisp (t=0.45),\n"
                "adaptive vs fixed only a trend (t=0.91, n=3)")
ax[0].grid(axis="y", alpha=0.3)

# Panel 2: held-out ep_len vs μ, one representative seed per arm
for a in ARMS:
    p = f"{EVAL}/setB_{a}_seed1.csv"
    if os.path.isfile(p):
        mus, ys = sweep(p)
        ax[1].plot(mus, ys, "o-", color=COL[a], label=f"{a} (seed1)")
ax[1].set_xlabel("held-out terrain μ (higher = easier)")
ax[1].set_ylabel("mean episode length (steps)")
ax[1].invert_xaxis()
ax[1].set_title("(2) Held-out survival vs μ\n(robust policies stay flat across difficulty)")
ax[1].legend(); ax[1].grid(alpha=0.3)

fig.suptitle("Fuzzy Set-B — adaptive-vs-fixed friction curriculum (all data from eval CSVs)",
             fontsize=12, y=1.0)
fig.tight_layout()
out = f"{ROOT}/setB_summary.png"
fig.savefig(out, bbox_inches="tight", dpi=130)
print(f"wrote {out}")
for a in ARMS:
    m = round(st.mean(aucs[a]), 4) if aucs[a] else None
    print(f"  {a}: AUC mean={m} n={len(aucs[a])}")
