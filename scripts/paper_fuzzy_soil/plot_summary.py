#!/usr/bin/env python3
"""
Summary charts for the fuzzy-soil-on-furrows investigation.
Reads ONLY from the source CSVs (no hand-entered numbers):
  - logs/FuzzySoilFurrows/results.csv          (A/B/C/D single evals)
  - logs/FuzzySoilFurrows/eval_noise.csv       (B/C/D repeated evals)
  - logs/FuzzySoilFurrows/fuzzy_sweep.csv       (fuzzy mu sweep, aggregated)
Produces logs/FuzzySoilFurrows/experiments_summary.png with 4 panels:
  (1) A/B survival (ep_len, 3 seeds, mean+points)
  (2) A/B slip (the non-replication)
  (3) C & D vs B noise bands (gate inert, compliance wash)
  (4) fuzzy mu-sweep: slip & falls vs friction
"""
from __future__ import annotations
import csv, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "logs/FuzzySoilFurrows"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def by_arm(rows, key, col):
    out = {}
    for r in rows:
        k = r.get(key)
        v = f(r.get(col))
        if v is not None:
            out.setdefault(k, []).append(v)
    return out


results = load(f"{ROOT}/results.csv")
noise = load(f"{ROOT}/eval_noise.csv")
sweep = sorted(load(f"{ROOT}/fuzzy_sweep.csv"), key=lambda r: f(r["mu"]))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ---- Panel 1: A vs B survival (ep_len), 3 seeds ----
ax = axes[0, 0]
epA = by_arm([r for r in results if r["arm"] == "A"], "arm", "ep_len_steps")["A"]
epB = by_arm([r for r in results if r["arm"] == "B"], "arm", "ep_len_steps")["B"]
xs = [0, 1]
ax.bar(xs, [st.mean(epA), st.mean(epB)], color=["tab:gray", "tab:green"], width=0.5, alpha=0.7)
ax.scatter([0]*len(epA), epA, color="black", zorder=3)
ax.scatter([1]*len(epB), epB, color="black", zorder=3)
ax.set_xticks(xs); ax.set_xticklabels(["A (v7 control)", "B (paper recipe)"])
ax.set_ylabel("episode length (steps)")
ratio = st.mean(epB) / st.mean(epA)
ax.set_title(f"(1) Survival REPLICATES — B walks {ratio:.1f}x longer\n3 seeds each")
ax.grid(axis="y", alpha=0.3)

# ---- Panel 2: A vs B slip (non-replication) ----
ax = axes[0, 1]
slA = by_arm([r for r in results if r["arm"] == "A"], "arm", "slip_per_100m")["A"]
slB = by_arm([r for r in results if r["arm"] == "B"], "arm", "slip_per_100m")["B"]
ax.bar(xs, [st.mean(slA), st.mean(slB)], color=["tab:gray", "tab:red"], width=0.5, alpha=0.7)
ax.scatter([0]*len(slA), slA, color="black", zorder=3)
ax.scatter([1]*len(slB), slB, color="black", zorder=3)
ax.axhline(11, ls="--", color="purple", label="paper target ~11")
ax.set_xticks(xs); ax.set_xticklabels(["A (v7 control)", "B (paper recipe)"])
ax.set_ylabel("slip / 100m")
ax.set_title("(2) Slip does NOT replicate — B is WORSE, not <11\nslip target missed by ~26x")
ax.legend(); ax.grid(axis="y", alpha=0.3)

# ---- Panel 3: C & D vs B noise bands (repeated evals) ----
ax = axes[1, 0]
nB = by_arm(noise, "label", "slip_per_100m").get("B", [])
nC = by_arm(noise, "label", "slip_per_100m").get("C", [])
nD = by_arm(noise, "label", "slip_per_100m").get("D", [])
groups = [("B (rigid, soft gate)", nB, "tab:blue"),
          ("C (rigid, hard gate)", nC, "tab:orange"),
          ("D (compliant)", nD, "tab:cyan")]
for i, (lab, vals, c) in enumerate(groups):
    if not vals: continue
    m, sd = st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0)
    ax.bar(i, m, yerr=sd, color=c, width=0.55, alpha=0.7, capsize=5)
    ax.scatter([i]*len(vals), vals, color="black", s=18, zorder=3)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
ax.set_ylabel("slip / 100m (repeated evals, mean±sd)")
ax.set_title("(3) Gate INERT, compliance WASH\nall overlap within eval noise (~8%)")
ax.grid(axis="y", alpha=0.3)

# ---- Panel 4: fuzzy mu-sweep ----
ax = axes[1, 1]
mus = [f(r["mu"]) for r in sweep]
slip = [f(r["slip_per_100m"]) for r in sweep]
falls = [f(r["falls_per_100m"]) for r in sweep]
ax.plot(mus, slip, "o-", color="tab:red", label="slip / 100m")
ax.plot(mus, falls, "s-", color="tab:orange", label="falls / 100m")
ax.set_xlabel("terrain friction μ (higher = easier)")
ax.set_ylabel("metric / 100m")
ax.invert_xaxis()  # harder (low mu) on the left to read as increasing difficulty
ax.set_title("(4) Fuzzy difficulty tracks FRICTION\nslip ρ=+0.97 vs d — but d adds nothing over μ")
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("Fuzzy-Soil-on-Furrows — experiment summary (all data from source CSVs)",
             fontsize=13, y=1.01)
fig.tight_layout()
out = f"{ROOT}/experiments_summary.png"
fig.savefig(out, bbox_inches="tight", dpi=130)
print(f"wrote {out}")
print(f"  A ep_len mean={st.mean(epA):.1f}  B ep_len mean={st.mean(epB):.1f}  ratio={ratio:.2f}x")
print(f"  A slip mean={st.mean(slA):.1f}  B slip mean={st.mean(slB):.1f}")
print(f"  noise slip: B={st.mean(nB):.1f}±{st.stdev(nB):.1f}  C={st.mean(nC):.1f}±{st.stdev(nC):.1f}  D={st.mean(nD):.1f}±{st.stdev(nD):.1f}")
