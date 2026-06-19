#!/usr/bin/env python3
"""Set D slide figure: 'fuzzy becomes a load-bearing DESCRIPTOR, not a better CURRICULUM'.

Two panels, honest-negative framing (Codex-vetted at the decision point):
  A) Descriptor — support-axis LOO-CV R^2 gain over friction mu:
       paper fuzzy 0.025 (blind) -> Set D fuzzy 0.416 (sees firmness),
       shaded raw/measured-physics ceiling band 0.334-0.417. v2 lands INSIDE
       the band: it MATCHES the ceiling, it does not beat raw.
  B) Curriculum proxy — productive fraction (higher=better), raw [mu,log2 K]
       vs Set D fuzzy [mu,d] (best 2-input form) at sensing noise sigma:
       pre-registered EQUIV; v2 never exceeds raw. Oracle as thin reference.

All numbers read from:
  docs/experiments/fuzzy_v2_loadbearing_result.md  (descriptor R^2 gains)
  docs/experiments/fuzzy_setC_dfh_result.md        (measured-physics ceiling)
  docs/experiments/fuzzy_v2_curriculum_plan.md     (2-input curriculum proxy)
Output: docs/slides/figs/setD_descriptor_vs_training.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "slides", "figs",
                                    "setD_descriptor_vs_training.png"))

# ---- data (real numbers only) -------------------------------------------------
# Panel A: support-axis LOO-CV R^2 gain over Y~mu
GAIN_PAPER = 0.025      # paper fuzzy d (support-blind)
GAIN_V2 = 0.416         # redesigned fuzzy d_v2 (load-bearing)
CEIL_LO = 0.334         # raw K (linear)
CEIL_HI = 0.417         # mu + measured sink_drag (nonlinear measured-physics)

# Panel B: curriculum proxy productive fraction (2-input steelman [mu, d_v2])
SIGMAS = ["0.0", "0.3", "0.4"]
RAW = [0.625, 0.494, 0.434]
V2 = [0.625, 0.472, 0.411]
ORACLE = 0.625

RED = "#c0392b"      # blind / bad
GREEN = "#2e9e54"    # the Set D fix (sees firmness)
BLUE = "#2c7fb8"     # raw soil numbers (neutral reference)
GREEN_B = "#74c476"  # v2 in panel B (muted, so a tie reads as a tie)
GRAY = "#7f7f7f"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5,
                     "axes.titleweight": "bold"})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.4))

# ---- Panel A: descriptor win --------------------------------------------------
xA = [0, 1]
barsA = axA.bar(xA, [GAIN_PAPER, GAIN_V2], width=0.56,
                color=[RED, GREEN], edgecolor="black", linewidth=0.7, zorder=3)
# ceiling band (NOT a target to beat)
axA.axhspan(CEIL_LO, CEIL_HI, color=GRAY, alpha=0.20, zorder=1)
axA.axhline(CEIL_LO, color=GRAY, lw=1.0, ls="--", zorder=2)
axA.axhline(CEIL_HI, color=GRAY, lw=1.0, ls="--", zorder=2)
axA.text(1.46, (CEIL_LO + CEIL_HI) / 2,
         "raw / measured-physics\nceiling  (0.33–0.42)",
         va="center", ha="center", fontsize=9.3, color="black",
         rotation=0,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, alpha=0.9))
# value + role labels
axA.text(0, GAIN_PAPER + 0.012, "0.025", ha="center", va="bottom",
         fontweight="bold", fontsize=11)
axA.text(0, -0.028, "paper fuzzy\n(blind to firmness)", ha="center", va="top",
         fontsize=9.5)
axA.text(1, GAIN_V2 + 0.012, "0.416", ha="center", va="bottom",
         fontweight="bold", fontsize=11)
axA.text(1, -0.028, "Set D fuzzy\n(sees firmness)", ha="center", va="top",
         fontsize=9.5)
# annotate: matches, does not exceed
axA.annotate("matches the ceiling —\ndoes NOT exceed raw",
             xy=(1, GAIN_V2), xytext=(0.30, 0.47),
             fontsize=9.2, ha="center", color="black",
             arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
axA.set_ylim(0, 0.52)
axA.set_xlim(-0.6, 2.05)
axA.set_xticks([])
axA.set_ylabel("support-axis LOO-CV $R^2$ gain over friction $\\mu$\n(higher = uses firmness more)",
               fontsize=9.8)
axA.set_title("As a DESCRIPTOR: the fix works")
axA.spines[["top", "right"]].set_visible(False)

# ---- Panel B: training tie ----------------------------------------------------
import numpy as np
xB = np.arange(len(SIGMAS))
w = 0.38
b1 = axB.bar(xB - w / 2, RAW, w, label="raw  $[\\mu,\\log_2 K]$",
             color=BLUE, edgecolor="black", linewidth=0.6, zorder=3)
b2 = axB.bar(xB + w / 2, V2, w, label="Set D fuzzy  $[\\mu,d]$ (best form)",
             color=GREEN_B, edgecolor="black", linewidth=0.6, zorder=3)
oracle_line = axB.axhline(ORACLE, color=GRAY, lw=1.4, ls=":", zorder=2,
                          label="oracle (best possible)")
for bars in (b1, b2):
    for r in bars:
        axB.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008,
                 f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=8.2)
axB.set_ylim(0, 0.72)
axB.set_xticks(xB)
axB.set_xticklabels([f"$\\sigma$={s}" for s in SIGMAS])
axB.set_xlabel("sensing noise $\\sigma$")
axB.set_ylabel("curriculum productive fraction\n(higher = better)", fontsize=9.8)
axB.set_title("As a TRAINING signal: only ties raw")
axB.legend(handles=[b1, b2, oracle_line], loc="upper right", fontsize=8.6, frameon=True)
axB.text(0.5, 0.045, "pre-registered EQUIVALENCE — v2 $\\approx$ raw, never exceeds",
         transform=axB.transAxes, ha="center", fontsize=9.0, style="italic",
         bbox=dict(boxstyle="round,pad=0.3", fc="#fff6e6", ec="#d9a441"))
axB.spines[["top", "right"]].set_visible(False)

fig.tight_layout(w_pad=2.5)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("wrote", OUT)
