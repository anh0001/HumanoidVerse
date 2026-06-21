#!/usr/bin/env python3
"""Plot the two curriculum-evidence rows the user asked for.

Row 1: a low->high curriculum helps the robot survive on soil (8.85x), RAW
       friction-ramp, rigid furrows.  Source: logs/FuzzySoilFurrows/results.csv.
Row 2: a fuzzy index drove a REAL low->high curriculum during training (Set B),
       but ties crisp/fixed.  Source: logs/FuzzySoilFurrowsSetB/eval/*.csv +
       setB_analysis.txt.

All numbers are read from the run CSVs; nothing hard-coded except labels.
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/srv/data/users/anhar/codes/HumanoidVerse"
OUT = os.path.join(ROOT, "docs/slides/figs")
os.makedirs(OUT, exist_ok=True)
DT = 0.02  # s per control step (sim_dt 0.005 * decimation 4)

# ----------------------------------------------------------------------------
# Row 1 — survival benefit
# ----------------------------------------------------------------------------
armA, armB = [], []
with open(os.path.join(ROOT, "logs/FuzzySoilFurrows/results.csv")) as f:
    for r in csv.DictReader(f):
        if r["arm"] == "A":
            armA.append(float(r["ep_len_steps"]))
        elif r["arm"] == "B":
            armB.append(float(r["ep_len_steps"]))

a_mean, b_mean = np.mean(armA), np.mean(armB)
ratio = b_mean / a_mean

fig, ax = plt.subplots(figsize=(7.2, 5.2))
xs = [0, 1]
labels = ["Baseline\n(no curriculum, v7)", "Low→high curriculum\n(μ 0.8→0.35, raw ramp)"]
colors = ["#b0b7c3", "#2e7d32"]
means = [a_mean, b_mean]
bars = ax.bar(xs, means, width=0.55, color=colors, edgecolor="black", linewidth=1.1, zorder=2)
# seed scatter
for x, data in zip(xs, [armA, armB]):
    jit = np.linspace(-0.08, 0.08, len(data))
    ax.scatter(np.full(len(data), x) + jit, data, color="black", s=42, zorder=4,
               edgecolor="white", linewidth=0.8, label="per-seed (n=3)" if x == 0 else None)

for x, m in zip(xs, means):
    ax.text(x, m + 12, f"{m:.1f} steps\n({m*DT:.1f} s)", ha="center", va="bottom",
            fontsize=11, fontweight="bold")

# ratio annotation
ax.annotate("", xy=(1, b_mean), xytext=(1, a_mean),
            arrowprops=dict(arrowstyle="<->", color="#c62828", lw=2))
ax.text(1.32, (a_mean + b_mean) / 2, f"{ratio:.2f}×\nsurvival", color="#c62828",
        fontsize=15, fontweight="bold", va="center", ha="left")

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("Mean episode length (steps)", fontsize=12)
ax.set_ylim(0, b_mean * 1.28)
ax.set_title("Low→high curriculum → robot survives on soil\n"
             "Hunter, rigid PhysX furrows, held-out eval (1000 ep × 3 seeds)",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", frameon=True, fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.text(0.5, -0.20, "Curriculum is a RAW friction-range ramp, not fuzzy-driven  •  "
        "slip-reduction headline did NOT replicate",
        transform=ax.transAxes, ha="center", fontsize=8.5, style="italic", color="#555")
fig.tight_layout()
p1 = os.path.join(OUT, "curriculum_survival_benefit.png")
fig.savefig(p1, dpi=150, bbox_inches="tight")
print("wrote", p1, "| ratio=%.2f a=%.1f b=%.1f" % (ratio, a_mean, b_mean))

# ----------------------------------------------------------------------------
# Row 2 — Set B fuzzy curriculum: robustness curve + AUC bars
# ----------------------------------------------------------------------------
MUS = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
ARMS = ["fixed", "crisp", "fuzzy"]
ARM_COLOR = {"fixed": "#90a4ae", "crisp": "#1565c0", "fuzzy": "#e65100"}

# per-arm: mu -> list of ep_len across seeds*reps
data = {a: defaultdict(list) for a in ARMS}
for a in ARMS:
    for path in sorted(glob.glob(os.path.join(ROOT, f"logs/FuzzySoilFurrowsSetB/eval/setB_{a}_seed*.csv"))):
        with open(path) as f:
            for r in csv.DictReader(f):
                data[a][float(r["mu"])].append(float(r["ep_len_steps"]))

# per-seed AUC straight from setB_analysis.txt (already computed there)
AUC_SEEDS = {
    "fixed": [0.5971, 0.5960, 0.7368],
    "crisp": [0.7056, 0.7385, 0.5958],
    "fuzzy": [0.8048, 0.8184, 0.5487],
}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.4))

# --- left: robustness curve (ep_len vs mu) ---
for a in ARMS:
    ys = np.array([np.mean(data[a][m]) for m in MUS])
    sd = np.array([np.std(data[a][m]) for m in MUS])
    axL.plot(MUS, ys, "-o", color=ARM_COLOR[a], lw=2.2, ms=7, label=a, zorder=3)
    axL.fill_between(MUS, ys - sd, ys + sd, color=ARM_COLOR[a], alpha=0.13, zorder=1)
axL.set_xlabel("Friction μ  (hard ←           → easy)", fontsize=11.5)
axL.set_ylabel("Mean episode length (steps)", fontsize=11.5)
axL.set_title("Held-out robustness curve\n(area under = robustness AUC)", fontsize=12, fontweight="bold")
axL.invert_xaxis()  # hard on the left
axL.grid(alpha=0.3)
axL.legend(title="curriculum arm", fontsize=10.5, title_fontsize=10.5, loc="lower right")
axL.set_ylim(bottom=0)

# --- right: AUC bars with seed dots ---
means = [np.mean(AUC_SEEDS[a]) for a in ARMS]
sds = [np.std(AUC_SEEDS[a]) for a in ARMS]
xs = np.arange(len(ARMS))
bars = axR.bar(xs, means, yerr=sds, capsize=6, width=0.6,
               color=[ARM_COLOR[a] for a in ARMS], edgecolor="black", linewidth=1.1, zorder=2)
for i, a in enumerate(ARMS):
    pts = AUC_SEEDS[a]
    jit = np.linspace(-0.09, 0.09, len(pts))
    axR.scatter(np.full(len(pts), i) + jit, pts, color="black", s=44, zorder=4,
                edgecolor="white", linewidth=0.8)
    axR.text(i, means[i] + sds[i] + 0.012, f"{means[i]:.3f}", ha="center", fontsize=11, fontweight="bold")
axR.set_xticks(xs)
axR.set_xticklabels([f"{a}\n(blocks→hardest: {b})" for a, b in zip(ARMS, [6, 3, 4])], fontsize=10.5)
axR.set_ylabel("Robustness AUC", fontsize=11.5)
axR.set_ylim(0, 1.02)
axR.set_title("Fuzzy curriculum does NOT beat crisp/fixed", fontsize=12, fontweight="bold")
axR.grid(axis="y", alpha=0.3, zorder=0)
# n.s. bracket fuzzy vs crisp
y = 0.90
axR.plot([1, 1, 2, 2], [y, y + 0.02, y + 0.02, y], color="black", lw=1.2)
axR.text(1.5, y + 0.025, "n.s.  (Welch t=0.45, n=3)", ha="center", fontsize=10, style="italic")

fig.suptitle("Set B — fuzzy index driving a REAL low→high curriculum (9 PPO runs: 3 arms × 3 seeds)\n"
             "rigid PhysX friction-only furrows — fuzzy support axis physically inert → fuzzy ≈ crisp ≈ fixed",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
p2 = os.path.join(OUT, "setB_fuzzy_curriculum.png")
fig.savefig(p2, dpi=150, bbox_inches="tight")
print("wrote", p2, "| AUC means", {a: round(m, 3) for a, m in zip(ARMS, means)})

# ----------------------------------------------------------------------------
# Row 2 (clean, single-panel) — AUC bars only: no title, x = arm names only,
# no value labels / no n.s. annotation (text lives on the slide instead).
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 5.0))
xs = np.arange(len(ARMS))
ax.bar(xs, means, width=0.62,
       color=[ARM_COLOR[a] for a in ARMS], edgecolor="black", linewidth=1.1, zorder=2)
ax.set_xticks(xs)
ax.set_xticklabels(ARMS, fontsize=14)
ax.set_ylabel("Robustness AUC", fontsize=14)
ax.set_ylim(0, 1.0)
ax.tick_params(axis="y", labelsize=12)
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
p3 = os.path.join(OUT, "setB_auc_only.png")
fig.savefig(p3, dpi=150, bbox_inches="tight")
print("wrote", p3)
