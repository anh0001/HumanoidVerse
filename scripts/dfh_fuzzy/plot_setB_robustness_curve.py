#!/usr/bin/env python3
"""Render the held-out robustness curve that the Set B AUC integrates.

survival = mean episode length / cap, vs friction mu, per curriculum arm.
AUC (per arm) = trapezoid(survival, mu) / mu-range  ==  area under this curve.
Reads the Set B held-out eval CSVs; writes the slide figure.
"""
import csv, glob, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/srv/data/users/anhar/codes/HumanoidVerse"
OUT = os.path.join(ROOT, "docs/slides/figs/setB_robustness_curve.png")
MUS = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
ARMS = ["fixed", "crisp", "fuzzy"]
COL = {"fixed": "#90a4ae", "crisp": "#1565c0", "fuzzy": "#e65100"}
CAP = 1000.0

data = {a: defaultdict(list) for a in ARMS}
for a in ARMS:
    for p in sorted(glob.glob(os.path.join(ROOT, f"logs/FuzzySoilFurrowsSetB/eval/setB_{a}_seed*.csv"))):
        with open(p) as f:
            for r in csv.DictReader(f):
                data[a][float(r["mu"])].append(float(r["ep_len_steps"]))

fig, ax = plt.subplots(figsize=(7.4, 4.7))
for a in ARMS:
    y = np.array([np.mean(data[a][m]) / CAP for m in MUS])
    ax.plot(MUS, y, "-o", color=COL[a], lw=2.3, ms=7, label=a, zorder=3)

yf = np.array([np.mean(data["fuzzy"][m]) / CAP for m in MUS])
ax.fill_between(MUS, 0, yf, color=COL["fuzzy"], alpha=0.15, zorder=1)
auc = np.trapz(yf, MUS) / (MUS[-1] - MUS[0])
ax.annotate(f"area ÷ μ-range = AUC ≈ {auc:.2f}\n(fuzzy)", xy=(0.55, 0.35),
            xytext=(0.45, 0.12), fontsize=11, color=COL["fuzzy"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COL["fuzzy"]))

ax.set_xlabel("friction μ   (hard ← 0.35 … 0.85 → easy)", fontsize=12)
ax.set_ylabel("survival  =  mean episode length ÷ cap", fontsize=12)
ax.set_title("The curve the AUC measures: survival vs soil difficulty (held-out)",
             fontsize=12.5, fontweight="bold")
ax.set_ylim(0, 1.0); ax.grid(alpha=0.3)
ax.legend(title="curriculum arm", fontsize=11, title_fontsize=11, loc="upper left")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT, "| AUCs:",
      {a: round(np.trapz([np.mean(data[a][m])/CAP for m in MUS], MUS)/(MUS[-1]-MUS[0]), 3) for a in ARMS})
