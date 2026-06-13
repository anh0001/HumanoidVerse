"""Generate learning-curve figures for the furrows/DFH slides.

Reads TensorBoard event files and writes PNGs into slides/figs/.
Run with the isaaclab python (has tensorboard + matplotlib).
"""
from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "/home/anhar/codes/HumanoidVerse"
OUT = os.path.join(ROOT, "extensions/dfh/slides/figs")
os.makedirs(OUT, exist_ok=True)

RUNS = {
    "v7": "logs/FixedStageFv7/20260525_060629-fixedFv7-locomotion-hunter",
    "rigid": "logs/DFH_Hunter_ROA/20260601_060616-DFH_paired_A_rigid_seed1-locomotion-hunter",
    "soil": "logs/DFH_Hunter_ROA/20260601_080848-DFH_paired_B_soil_seed1-locomotion-hunter",
    "s1": "logs/DFH_Hunter_ROA/20260526_022909-DFH_S1_from_v7_seed1-locomotion-hunter",
    "s2": "logs/DFH_Hunter_ROA/20260526_085351-DFH_S2_from_S1_seed1-locomotion-hunter",
    "s3": "logs/DFH_Hunter_ROA/20260526_201433-DFH_S3_from_S2_seed1-locomotion-hunter",
}

DT = 0.02  # control timestep (s); ~1000 steps = 20 s episode cap


def load(run_key: str, tag: str):
    ev = sorted(glob.glob(os.path.join(ROOT, RUNS[run_key]) + "/events.out.tfevents*"))[-1]
    ea = EventAccumulator(ev, size_guidance={"scalars": 0})
    ea.Reload()
    s = ea.Scalars(tag)
    return np.array([p.step for p in s], float), np.array([p.value for p in s], float)


def smooth(y: np.ndarray, w: int = 25) -> np.ndarray:
    if len(y) < w:
        return y
    k = np.ones(w) / w
    pad = np.r_[np.full(w, y[0]), y, np.full(w, y[-1])]
    return np.convolve(pad, k, mode="same")[w:-w]


plt.rcParams.update({"font.size": 13, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})

C_REW = "#1f7a3d"
C_RIGID = "#2166ac"
C_SOIL = "#c0392b"
C_DRAG = "#8e44ad"


def raw_smooth(ax, x, y, color, label, lw=2.4):
    ax.plot(x, y, color=color, alpha=0.18, lw=1)
    ax.plot(x, smooth(y), color=color, lw=lw, label=label)


# ---------------------------------------------------------------- Fig 1
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
x, r = load("v7", "Train/mean_reward")
raw_smooth(ax[0], x, r, C_REW, "mean reward")
ax[0].set_title("Reward climbs and plateaus")
ax[0].set_xlabel("PPO iteration")
ax[0].set_ylabel("mean episode reward")
ax[0].axhline(0, color="k", lw=0.8, alpha=0.4)
ax[0].legend(loc="lower right", frameon=False)

x, e = load("v7", "Train/mean_episode_length")
raw_smooth(ax[1], x, e, C_RIGID, "episode length")
ax[1].axhline(1000, color="k", ls="--", lw=1, alpha=0.6)
ax[1].text(x[len(x) // 2], 1010, "20 s episode cap (~1000 steps)", fontsize=10, alpha=0.7)
ax[1].set_title("Robot survives the full episode")
ax[1].set_xlabel("PPO iteration")
ax[1].set_ylabel("episode length (sim steps)")
secax = ax[1].secondary_yaxis("right", functions=(lambda v: v * DT, lambda v: v / DT))
secax.set_ylabel("seconds")
fig.suptitle("Experiment 1 — Rigid furrows: the walker learns (deployment model v7)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_furrows_learn.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 2
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
for key, c, lab in [("rigid", C_RIGID, "RIGID-trained"), ("soil", C_SOIL, "SOIL-trained (DFH)")]:
    x, r = load(key, "Train/mean_reward")
    raw_smooth(ax[0], x, r, c, lab)
ax[0].set_title("Both reach the same reward")
ax[0].set_xlabel("PPO iteration")
ax[0].set_ylabel("mean episode reward")
ax[0].legend(loc="lower right", frameon=False)

for key, c, lab in [("rigid", C_RIGID, "RIGID-trained"), ("soil", C_SOIL, "SOIL-trained (DFH)")]:
    x, e = load(key, "Train/mean_episode_length")
    raw_smooth(ax[1], x, e, c, lab)
ax[1].axhline(1000, color="k", ls="--", lw=1, alpha=0.6)
ax[1].set_title("Both survive full episodes")
ax[1].set_xlabel("PPO iteration")
ax[1].set_ylabel("episode length (sim steps)")
ax[1].legend(loc="lower right", frameon=False)
fig.suptitle("Experiment 2 — Both policies learn equally well; soil training adds no edge", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_paired_learn.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 3
fig, ax = plt.subplots(figsize=(8.5, 4.2))
x, p = load("soil", "Env/dfh_applied_drag_pct_weight")
raw_smooth(ax, x, p * 100, C_DRAG, "drag on the robot")
ax.axhspan(5, 10, color="green", alpha=0.10)
ax.text(x[len(x) // 3], 6.0, "5-10% body-weight design band", fontsize=11, color="green", alpha=0.9)
ax.set_ylim(0, 14)
ax.set_xlabel("PPO iteration")
ax.set_ylabel("soil drag as % of body weight")
ax.legend(loc="lower right", frameon=False)
fig.suptitle("Experiment 2 — A real load (~12% body weight) the robot learns to walk through",
             fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_drag_budget.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 4 (difficulty)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax2 = ax.twinx()
offset = 0
labels = []
for key, name in [("s1", "Stage 1 (easy soil)"), ("s2", "Stage 2 (medium)"), ("s3", "Stage 3 (full/wet)")]:
    xe, e = load(key, "Train/mean_episode_length")
    xp, p = load(key, "Env/dfh_applied_drag_pct_weight")
    ax.plot(xe, smooth(e), color=C_RIGID, lw=2.4)
    ax2.plot(xp, smooth(p * 100), color=C_DRAG, lw=2.4, ls="--")
ax.set_xlabel("PPO iteration (curriculum proceeds left -> right)")
ax.set_ylabel("episode length (sim steps)", color=C_RIGID)
ax.tick_params(axis="y", labelcolor=C_RIGID)
ax2.set_ylabel("drag as % of body weight", color=C_DRAG)
ax2.tick_params(axis="y", labelcolor=C_DRAG)
ax.axhline(1000, color="k", ls=":", lw=1, alpha=0.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], color=C_RIGID, lw=2.4, label="episode length (solid)"),
                   Line2D([0], [0], color=C_DRAG, lw=2.4, ls="--", label="drag % body weight (dashed)")],
          loc="center right", frameon=False)
fig.suptitle("Why deeper soil is brutal: as drag rises (8% -> 30% BW), survival collapses", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_difficulty.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 5 + 6 (reward decomposition)
def rew_terms(run_key):
    ev = sorted(glob.glob(os.path.join(ROOT, RUNS[run_key]) + "/events.out.tfevents*"))[-1]
    ea = EventAccumulator(ev, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for t in ea.Tags()["scalars"]:
        if t.startswith("Episode/rew_"):
            s = ea.Scalars(t)
            out[t.replace("Episode/rew_", "")] = (np.array([p.step for p in s], float),
                                                  np.array([p.value for p in s], float))
    return out

terms = rew_terms("v7")
finals = {k: smooth(v[1])[-1] for k, v in terms.items()}
order = sorted(finals, key=lambda k: finals[k])  # ascending for barh

fig, ax = plt.subplots(figsize=(10, 5.2))
vals = [finals[k] for k in order]
colors = [C_REW if v >= 0 else C_SOIL for v in vals]
ax.barh(range(len(order)), vals, color=colors)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=10)
ax.axvline(0, color="k", lw=0.8)
for i, v in enumerate(vals):
    ax.text(v + (0.06 if v >= 0 else -0.06), i, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.set_xlim(-0.6, 3.9)
ax.set_xlabel("contribution to per-step reward (final policy)")
ax.set_title("What the policy is paid for: velocity tracking dominates", fontsize=14)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_REW, label="reward (good behaviour)"),
                   Patch(color=C_SOIL, label="penalty (bad behaviour)")],
          loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_reward_budget.png"), bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.6))
top = ["tracking_lin_vel", "tracking_ang_vel", "gait_phase",
       "penalty_action_rate", "penalty_feet_contact_forces"]
palette = {"tracking_lin_vel": "#1f7a3d", "tracking_ang_vel": "#2166ac",
           "gait_phase": "#e08214", "penalty_action_rate": "#c0392b",
           "penalty_feet_contact_forces": "#8e44ad"}
for k in top:
    x, y = terms[k]
    ax.plot(x, smooth(y), color=palette[k], lw=2.4, label=k)
ax.axhline(0, color="k", lw=0.8, alpha=0.4)
ax.set_xlabel("PPO iteration")
ax.set_ylabel("reward contribution")
ax.set_title("How the top terms grew as the robot learned", fontsize=14)
ax.legend(loc="upper left", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_reward_growth.png"), bbox_inches="tight")
plt.close(fig)

print("wrote:", sorted(os.listdir(OUT)))
