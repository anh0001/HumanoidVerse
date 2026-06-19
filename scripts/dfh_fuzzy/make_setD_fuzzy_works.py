#!/usr/bin/env python3
"""Set D positive slide: 'the redesigned fuzzy index predicts real soil difficulty'.

Codex-vetted honest-positive chart (Option B): scatter of the redesigned fuzzy
difficulty d_v2 (x) vs the MEASURED controller difficulty Y (y), one point per
measured walking cell, colored by firmness K and shaped by traction mu. Shows the
fuzzy descriptor orders real difficulty and separates firmness -- WITHOUT comparing
to raw and WITHOUT drawing any surface region unsupported by data.

Guardrails (Codex):
  - only the measured walking cells (no full-breakpoint surface / no extrapolation),
  - monotone/rank trend, not a high-order fit,
  - annotate partial Spearman(Y, d_v2 | mu) (isolates the firmness axis),
  - print a 'descriptor-validation-only' disclaimer on the slide (in the .tex).

Numbers computed live from the real controller + data; cross-checked vs
fuzzy_v2_loadbearing_result.md (partial Spearman +0.715, Spearman(d_v2,K) -0.916).

Output: docs/slides/figs/setD_fuzzy_predicts.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from fuzzy_v2_loadbearing import fuzzy_d_v2  # noqa: E402
from analyze_dfh_fuzzy import (  # noqa: E402
    K_to_kn, walking_mask, load, rankdata, spearman, partial_spearman,
)

CSV = os.path.join(ROOT, "logs", "DFH_fuzzy", "sweep_rigid.csv")
OUT = os.path.join(ROOT, "docs", "slides", "figs", "setD_fuzzy_predicts.png")

# ---- data (real controller + measured sweep) ---------------------------------
rows = load(CSV)
walk = [r for r, ok in zip(rows, walking_mask(rows)) if ok]
mu = np.array([float(r["mu"]) for r in walk])
K = np.array([float(r["K"]) for r in walk])
kn = np.array([K_to_kn(k) for k in K])
d_v2 = np.array([fuzzy_d_v2(m, kk) for m, kk in zip(mu, kn)])
Y = (rankdata([float(r["vel_err"]) for r in walk])
     + rankdata([float(r["slip_per_100m"]) for r in walk])
     + rankdata([float(r["falls_per_100m"]) for r in walk])
     - rankdata([float(r["distance_m"]) for r in walk]))

ps = partial_spearman(Y, d_v2, mu)   # firmness-axis stat (expect +0.715)
sp = spearman(Y, d_v2)               # overall ordering
sdk = spearman(d_v2, K)              # controller vs firmness (expect -0.916)
print(f"n={len(walk)}  partial Spearman(Y,d_v2|mu)={ps:+.3f}  "
      f"overall Spearman(Y,d_v2)={sp:+.3f}  Spearman(d_v2,K)={sdk:+.3f}")

# ---- plot --------------------------------------------------------------------
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(7.4, 4.8))

# color = firmness K (log), shape = traction mu; trend drawn WITHIN each traction
# level (low-order degree-1) so the visual matches the cited partial Spearman.
MU_LEVELS = sorted(set(np.round(mu, 2)))
MARKERS = {MU_LEVELS[0]: "o", MU_LEVELS[1]: "s", MU_LEVELS[2]: "^"}
sc = None
for m in MU_LEVELS:
    sel = np.round(mu, 2) == m
    sc = ax.scatter(d_v2[sel], Y[sel], c=np.log2(K[sel]), cmap="viridis_r",
                    vmin=0, vmax=4, marker=MARKERS[m], s=85,
                    edgecolor="black", linewidth=0.5, zorder=3)
    b1, b0 = np.polyfit(d_v2[sel], Y[sel], 1)
    xs = np.linspace(d_v2[sel].min(), d_v2[sel].max(), 20)
    ax.plot(xs, b1 * xs + b0, color="0.5", ls="--", lw=1.0, alpha=0.65, zorder=1)

cbar = fig.colorbar(sc, ax=ax, ticks=[0, 1, 2, 3, 4], pad=0.02)
cbar.ax.set_yticklabels(["1", "2", "4", "8", "16"])
cbar.set_label("soil firmness $K$   (softer $\\rightarrow$ firmer)")

ax.set_xlabel("redesigned fuzzy difficulty $d_{v2}$  (higher = harder)")
ax.set_ylabel("measured difficulty $Y$  (rank index, higher = harder)")
ax.set_title("The redesigned fuzzy index predicts real soil difficulty",
             fontweight="bold", fontsize=12.5)

# stat annotation (the firmness-axis result)
ax.text(0.035, 0.965,
        f"partial Spearman$(Y, d_{{v2}} \\mid \\mu) = {ps:+.2f}$\n"
        f"(controls for traction $\\rightarrow$ isolates firmness)",
        transform=ax.transAxes, va="top", ha="left", fontsize=9.6,
        bbox=dict(boxstyle="round,pad=0.4", fc="#eaf5ea", ec="#2e9e54"))

# legend for traction marker shapes
handles = [Line2D([0], [0], marker=MARKERS[m], color="0.3", ls="none",
                  markersize=8, markeredgecolor="black",
                  label=f"$\\mu$ = {m:.2f}") for m in MU_LEVELS]
ax.legend(handles=handles + [Line2D([0], [0], color="0.5", ls="--", lw=1.0,
          label="trend within $\\mu$")], loc="lower right", fontsize=8.8,
          frameon=True, title="traction")

ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("wrote", OUT)
