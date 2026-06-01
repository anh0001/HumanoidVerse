#!/usr/bin/env python3
"""
DFH fuzzy descriptor sweep (Codex design) — frozen-policy, NO training.

Evaluates the mild-soil DFH walker across a 2-D soil grid (traction mu x Bekker
support stiffness K) and records task difficulty + DFH force diagnostics per cell.
Purpose: in a DEFORMABLE-soil sim (DFH), is the fuzzy index's SUPPORT axis still
inert (as in rigid PhysX, Set A/B) or does it now move difficulty — and does the
fuzzy index d add information over raw mu? (see docs/experiments/fuzzy_setC_plan.md)

Holds everything at the walker's training DFH config EXCEPT the two swept axes:
  - traction:  mu_along = mu_across = mu          (isotropic; the friction axis)
  - support:   kc = 1400*K, k_phi = 820000*K      (Bekker pressure-sinkage stiffness)
force_coupling on (off = SHADOW negative control), writeback off (avoids the shared
terrain-prim cross-env bug), eval_match_train default True.

Each (mu, K, seed) is its own eval run (no per-env DFH randomization). Parses task
metrics + "DFH_METRIC <k>: <v>" diagnostics from sample_eps stdout into one CSV.
"""
from __future__ import annotations
import argparse, csv, os, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PY = os.environ.get("PYTHON", "/srv/data/users/anhar/miniconda3/envs/isaaclab/bin/python")
KC0, KPHI0 = 1400.0, 820000.0   # Bekker defaults (config.py); scaled by K

# task-metric stdout labels -> csv field
TASK_PATTERNS = {
    "ep_len_steps":   r"Average episode length:\s*([0-9.]+)",
    "distance_m":     r"Average distance per episode:\s*([0-9.]+)",
    "falls_per_100m": r"Falls per 100m:\s*([0-9.]+)",
    "slip_per_100m":  r"Slip distance per 100m:\s*([0-9.]+)",
    "vel_err":        r"Velocity error \(m/s\):\s*([0-9.]+)",
}
# DFH diagnostics we keep (subset of DFH_METRIC lines)
DFH_KEEP = [
    "dfh_mean_sink_m", "dfh_max_sink_m", "dfh_mean_sink_drag_n", "dfh_mean_aniso_drag_n",
    "dfh_mean_applied_drag_n", "dfh_total_drag_per_robot_n", "dfh_applied_drag_pct_weight",
    "dfh_drag_clipped_frac", "dfh_mean_normal_force_n", "dfh_stance_fraction",
    "dfh_force_coupling_on", "dfh_attached",
]


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def parse_eval(text):
    text = strip_ansi(text)
    row = {}
    for field, pat in TASK_PATTERNS.items():
        m = re.search(pat, text)
        row[field] = float(m.group(1)) if m else None
    dfh = dict(re.findall(r"DFH_METRIC (\w+):\s*([-0-9.]+)", text))
    for k in DFH_KEEP:
        row[k] = float(dfh[k]) if k in dfh else None
    return row


def build_cmd(ckpt, mu, K, seed, command, num_envs, num_eps, force_coupling):
    kc, kphi = KC0 * K, KPHI0 * K
    return [
        PY, "humanoidverse/sample_eps.py",
        f"+checkpoint={ckpt}",
        "+terrain=terrain_dfh_stage1_easy",
        f"+eval_command={command}",
        f"+seed={seed}",
        f"num_envs={num_envs}", f"++simulator.config.scene.num_envs={num_envs}",
        f"+num_episodes={num_eps}", "headless=True",
        # swept axes
        f"++terrain.dfh.params.anisotropy.mu_along={mu}",
        f"++terrain.dfh.params.anisotropy.mu_across={mu}",
        f"++terrain.dfh.params.bekker.kc={kc}",
        f"++terrain.dfh.params.bekker.k_phi={kphi}",
        # held fixed at walker-training values
        "++terrain.dfh.force_coupling.sinkage_drag_k=8.0",
        "++terrain.dfh.params.sinkage_floor_m=-0.05",
        "++terrain.dfh.writeback_enabled=False",
        f"++terrain.dfh.force_coupling.enabled={force_coupling}",
    ]


def make_env():
    env = dict(os.environ)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    for v in ("CARB_APP_PATH", "EXP_PATH", "ISAAC_PATH", "ISAACSIM_PATH"):
        env.pop(v, None)          # let pip-isaacsim self-resolve (else IApp acquire fails)
    env["ISAACLAB_PATH"] = str(REPO / "IsaacLab")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="logs/DFH_hf_models/walkers/mildsoil_walker/model_5050.pt")
    ap.add_argument("--mus", default="0.50,0.56,0.62")
    ap.add_argument("--Ks", default="0.80,1.00,1.25")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--command", default="[0.3,0.0,0.0]")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-eps", type=int, default=50)
    ap.add_argument("--force-coupling", default="True", choices=["True", "False"])
    ap.add_argument("--out-csv", default="logs/DFH_fuzzy/sweep.csv")
    ap.add_argument("--tag", default="grid")
    args = ap.parse_args()

    os.chdir(REPO)
    ckpt = args.policy
    if not os.path.isfile(ckpt):
        print(f"[dfh-sweep] FATAL policy missing: {ckpt}", file=sys.stderr); return 2
    mus = [float(x) for x in args.mus.split(",")]
    Ks = [float(x) for x in args.Ks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    logdir = out.parent / f"logs_{args.tag}"; logdir.mkdir(parents=True, exist_ok=True)
    env = make_env()

    fields = (["mu", "K", "kc", "k_phi", "seed", "force_coupling"]
              + list(TASK_PATTERNS.keys()) + DFH_KEEP)
    rows = []
    total = len(mus) * len(Ks) * len(seeds)
    i = 0
    for mu in mus:
        for K in Ks:
            for seed in seeds:
                i += 1
                cell = f"mu{mu}_K{K}_s{seed}_fc{args.force_coupling}"
                logf = logdir / f"{cell}.log"
                print(f"[dfh-sweep] ({i}/{total}) {cell} -> running", flush=True)
                cmd = build_cmd(ckpt, mu, K, seed, args.command,
                                args.num_envs, args.num_eps, args.force_coupling)
                with open(logf, "w") as lf:
                    subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
                text = logf.read_text(errors="replace")
                row = parse_eval(text)
                row.update({"mu": mu, "K": K, "kc": KC0 * K, "k_phi": KPHI0 * K,
                            "seed": seed, "force_coupling": args.force_coupling})
                rows.append(row)
                print(f"[dfh-sweep]   ep_len={row.get('ep_len_steps')} "
                      f"vel_err={row.get('vel_err')} falls100={row.get('falls_per_100m')} "
                      f"sink_m={row.get('dfh_mean_sink_m')} "
                      f"sink_drag_n={row.get('dfh_mean_sink_drag_n')} "
                      f"clip={row.get('dfh_drag_clipped_frac')}", flush=True)
                # write incrementally
                with open(out, "w", newline="") as cf:
                    w = csv.DictWriter(cf, fieldnames=fields)
                    w.writeheader()
                    for r in rows:
                        w.writerow({k: r.get(k) for k in fields})
    print(f"[dfh-sweep] DONE {len(rows)} cells -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
