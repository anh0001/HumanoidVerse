#!/usr/bin/env python3
"""Generate the two headline DFH figures for the slide deck (reads sweep_rigid.csv)."""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze_dfh_fuzzy import walking_mask, rankdata, loo_cv_r2, fuzzy_d, K_to_kn

OUT = "docs/slides/figs"; os.makedirs(OUT, exist_ok=True)
rows = []
for r in csv.DictReader(open("logs/DFH_fuzzy/sweep_rigid.csv")):
    d = {}
    for k, v in r.items():
        try: d[k] = float(v)
        except (TypeError, ValueError): d[k] = v
    rows.append(d)

# ---- Figure 1: support axis — sinkage & sink-drag vs K (all cells, mean per K) ----
Ks = sorted(set(r["K"] for r in rows))
sink = [1000*np.mean([r["dfh_mean_sink_m"] for r in rows if r["K"]==k]) for k in Ks]  # mm
drag = [np.mean([r["dfh_mean_sink_drag_n"] for r in rows if r["K"]==k]) for k in Ks]
fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
c1, c2 = "tab:blue", "tab:red"
ax1.plot(Ks, sink, "o-", color=c1, lw=2, ms=8, label="foot sinkage")
ax1.set_xscale("log", base=2); ax1.set_xticks(Ks); ax1.set_xticklabels([str(int(k)) for k in Ks])
ax1.set_xlabel("soil firmness  K  (Bekker stiffness scale; higher = firmer)")
ax1.set_ylabel("mean foot sinkage  (mm)", color=c1); ax1.tick_params(axis="y", labelcolor=c1)
ax2 = ax1.twinx()
ax2.plot(Ks, drag, "s--", color=c2, lw=2, ms=8, label="sideways drag")
ax2.set_ylabel("mean sideways soil drag  (N)", color=c2); ax2.tick_params(axis="y", labelcolor=c2)
ax1.set_title("DFH deformable soil: the SUPPORT axis is real\n(firmer soil → feet sink less → less drag — impossible in rigid PhysX)")
ax1.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/dfh_support_axis.png", dpi=140, bbox_inches="tight")
print("wrote dfh_support_axis.png  sink(mm)=", [round(x,1) for x in sink])

# ---- Figure 2: does it help predict difficulty? CV-R^2 gain over mu (walking cells) ----
walk = [r for r, ok in zip(rows, walking_mask(rows)) if ok]
def col(k): return np.array([r[k] for r in walk], float)
Y = (rankdata(col("vel_err")) + rankdata(col("slip_per_100m"))
     + rankdata(col("falls_per_100m")) - rankdata(col("distance_m")))
mu = col("mu"); K = col("K"); sinkm = col("dfh_mean_sink_m"); sdrag = col("dfh_mean_sink_drag_n")
dd = np.array([fuzzy_d(m, K_to_kn(k)) for m, k in zip(mu, K)])
base = loo_cv_r2(Y, mu.reshape(-1, 1))
labels = ["raw K", "measured\nsinkage", "measured\ndrag", "paper\nFUZZY index"]
gains = [loo_cv_r2(Y, np.c_[mu, x]) - base for x in (K, sinkm, sdrag, dd)]
cols = ["tab:green", "tab:green", "tab:green", "tab:red"]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
b = ax.bar(range(len(gains)), gains, color=cols, alpha=0.8)
ax.axhline(0.10, ls="--", color="gray", label="“adds real value” bar (+0.10)")
ax.set_xticks(range(len(gains))); ax.set_xticklabels(labels)
ax.set_ylabel("extra difficulty-prediction power\nover raw friction μ  (CV-R² gain)")
ax.set_title("Where both soil axes are physical, raw soil numbers predict difficulty —\nthe paper's FUZZY index adds almost nothing")
for i, g in enumerate(gains):
    ax.text(i, g + 0.012, f"+{g:.2f}", ha="center", fontweight="bold")
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3); ax.set_ylim(min(0, min(gains)-0.05), max(gains)+0.1)
fig.tight_layout(); fig.savefig(f"{OUT}/dfh_fuzzy_vs_raw.png", dpi=140, bbox_inches="tight")
print("wrote dfh_fuzzy_vs_raw.png  gains=", [round(g,3) for g in gains])
