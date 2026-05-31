#!/usr/bin/env python3
"""
Set B orchestrator — block-chained global ground-friction curriculum.

One invocation = ONE arm x ONE seed = an N-block checkpoint-chained PPO-ROA run.
Each block is a fresh train_agent.py job built at the controller-chosen global mu
(++terrain.static_friction=mu ++terrain.dynamic_friction=mu — the construction-time
knob validated in Set A), warm-started from the previous block's checkpoint with
optimizer/RNG state resumed (load_optimizer=True) for continuity.

The 3 arms (fixed / crisp / fuzzy) share identical ladder, total iterations, signal,
DR, reward, obs, env — they differ ONLY in how the next block's mu is chosen
(curriculum_controller.py). After each block we read Train/mean_episode_length from
that block's TensorBoard log and feed it to the controller.

Matched compute: blocks * iters_per_block identical across all arms/seeds.

Usage (via run_arm.sh which sets the Isaac env), e.g.:
  python scripts/setB_fuzzy_curriculum/run_block_chain.py \
    --arm fuzzy --seed 1 --blocks 8 --iters-per-block 94 --num-envs 2048
Smoke/pre-flight (tiny, CPU-cheap-ish):
  python ... --arm crisp --seed 1 --blocks 2 --iters-per-block 4 --num-envs 64 --smoke
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, glob
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from curriculum_controller import CurriculumController  # noqa: E402

PROJECT = "FuzzySoilFurrowsSetB"


def find_run_dir(project: str, exp_name: str):
    """Newest logs/<project>/*<exp_name>* run directory, or None."""
    cands = sorted(glob.glob(str(REPO / "logs" / project / f"*{exp_name}*")),
                   key=lambda p: os.path.getmtime(p), reverse=True)
    cands = [c for c in cands if os.path.isdir(c)]
    return cands[0] if cands else None


def latest_ckpt(run_dir: str):
    """Highest-iteration model_*.pt under run_dir, or None."""
    best, best_it = None, -1
    for p in glob.glob(os.path.join(run_dir, "**", "model_*.pt"), recursive=True):
        try:
            it = int(os.path.basename(p).split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        if it > best_it:
            best, best_it = p, it
    return best


def read_mean_ep_len(run_dir: str):
    """Last value of Train/mean_episode_length from this run's TensorBoard log."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return None
    ea = event_accumulator.EventAccumulator(
        run_dir, size_guidance={event_accumulator.SCALARS: 0})
    try:
        ea.Reload()
    except Exception:
        return None
    tag = "Train/mean_episode_length"
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    return float(events[-1].value) if events else None


def build_block_cmd(py, arm, seed, block, mu, iters, num_envs, warm_ckpt,
                    load_optimizer, exp_name, smoke):
    """train_agent.py command for one block (mirrors Arm B common_args)."""
    cmd = [
        py, "humanoidverse/train_agent.py",
        "+exp=locomotion", "algo=ppo_roa",
        "+rewards=loco/reward_hunter_paper_soil",
        "+robot=hunter/hunter", "+simulator=isaacsim",
        "+terrain=terrain_furrows_stage1_easy",
        "+obs=loco/leggedloco_obs_history_wolinvel",
        "+domain_rand=DR_paper_S3",
        f"num_envs={num_envs}", f"seed={seed}", "headless=True",
        "++env.config.env_spacing=2.5",
        f"++algo.config.num_learning_iterations={iters}",
        f"++algo.config.save_interval={max(1, iters // 2)}",
        f"++algo.config.load_optimizer={load_optimizer}",
        "++algo.config.actor_learning_rate=2.5e-4",
        "++algo.config.critic_learning_rate=2.5e-4",
        "++algo.config.desired_kl=0.005",
        "++algo.config.entropy_coef=0.001",
        "++algo.config.clip_param=0.10",
        "++env.config.termination.terminate_by_contact=True",
        "++rewards.reward_scales.tracking_lin_vel=4.0",
        "++env.config.locomotion_command_ranges.lin_vel_x=[0.25,0.45]",
        "++env.config.locomotion_command_ranges.lin_vel_y=[0.0,0.0]",
        "++env.config.locomotion_command_ranges.ang_vel_yaw=[0.0,0.0]",
        "++env.config.reset_randomization.enable=True",
        # --- the curriculum knob: global ground traction mu (Set A's proven axis) ---
        f"++terrain.static_friction={mu}",
        f"++terrain.dynamic_friction={mu}",
        f"++checkpoint={warm_ckpt}",
        f"project_name={PROJECT}",
        f"experiment_name={exp_name}",
    ]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["fixed", "crisp", "fuzzy"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--iters-per-block", type=int, default=94)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--ladder", default="0.80,0.65,0.50,0.35",
                    help="friction rungs EASY->HARD, comma-separated")
    ap.add_argument("--ep-len-cap", type=float, default=1000.0)
    ap.add_argument("--crisp-threshold", type=float, default=0.50)
    ap.add_argument("--fuzzy-breaks", default="0.35,0.50,0.65",
                    help="fuzzy mastery breakpoints lo,mid,hi")
    ap.add_argument("--warm-ckpt",
                    default="logs/FixedStageFv7/20260525_060629-fixedFv7-locomotion-hunter/model_4250.pt")
    ap.add_argument("--out-root", default="logs/FuzzySoilFurrowsSetB")
    ap.add_argument("--py", default=os.environ.get(
        "PYTHON", "/srv/data/users/anhar/miniconda3/envs/isaaclab/bin/python"))
    ap.add_argument("--smoke", action="store_true",
                    help="pre-flight functional test: tiny blocks, no failure on small metrics")
    args = ap.parse_args()

    os.chdir(REPO)
    rungs = [float(x) for x in args.ladder.split(",")]
    flo, fmid, fhi = (float(x) for x in args.fuzzy_breaks.split(","))
    warm = args.warm_ckpt
    if not os.path.isfile(warm):
        print(f"[setB] FATAL warm checkpoint missing: {warm}", file=sys.stderr)
        return 2

    tag = f"setB_{args.arm}_seed{args.seed}"
    out_dir = Path(args.out_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_csv = out_dir / "decisions.csv"
    manifest_json = out_dir / "manifest.json"

    ctrl = CurriculumController(
        rungs=rungs, arm=args.arm, total_blocks=args.blocks,
        ep_len_cap=args.ep_len_cap, crisp_threshold=args.crisp_threshold,
        fuzzy_lo=flo, fuzzy_mid=fmid, fuzzy_hi=fhi,
    )

    print(f"[setB] arm={args.arm} seed={args.seed} blocks={args.blocks} "
          f"iters/block={args.iters_per_block} ladder={rungs} smoke={args.smoke}")
    print(f"[setB] total iters = {args.blocks * args.iters_per_block} (matched across arms)")

    chain_ckpt = warm
    block_records = []
    for block in range(args.blocks):
        mu = ctrl.current_mu
        load_opt = (block > 0)  # block 0 warms from a foreign ckpt -> no optimizer load
        exp_name = f"{tag}_b{block}_mu{mu:.2f}"
        cmd = build_block_cmd(args.py, args.arm, args.seed, block, mu,
                              args.iters_per_block, args.num_envs, chain_ckpt,
                              load_opt, exp_name, args.smoke)
        print(f"\n[setB] === BLOCK {block} mu={mu:.2f} warm={os.path.basename(chain_ckpt)} "
              f"load_opt={load_opt} ===")
        block_log = out_dir / f"block{block}_mu{mu:.2f}.log"
        with open(block_log, "w") as lf:
            ret = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        if ret != 0:
            print(f"[setB] FATAL block {block} train_agent.py exited {ret}; see {block_log}",
                  file=sys.stderr)
            return 3

        run_dir = find_run_dir(PROJECT, exp_name)
        if not run_dir:
            print(f"[setB] FATAL no run dir for {exp_name}", file=sys.stderr)
            return 4
        new_ckpt = latest_ckpt(run_dir)
        if not new_ckpt:
            print(f"[setB] FATAL no checkpoint in {run_dir}", file=sys.stderr)
            return 5
        ep_len = read_mean_ep_len(run_dir)
        print(f"[setB] block {block}: run_dir={run_dir}")
        print(f"[setB] block {block}: mean_ep_len={ep_len} ckpt={os.path.basename(new_ckpt)}")

        rec = ctrl.decide_next(block, ep_len)
        rec.update({"run_dir": run_dir, "ckpt": new_ckpt, "mu_trained": mu})
        block_records.append(rec)
        print(f"[setB] block {block}: mastery={rec['mastery']} advanced={rec['advanced']} "
              f"-> next mu={rec['next_mu']:.2f}")

        # persist incrementally so a crash still leaves a readable trail
        with open(decisions_csv, "w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(block_records[0].keys()))
            w.writeheader(); w.writerows(block_records)
        chain_ckpt = new_ckpt

    manifest = {
        "arm": args.arm, "seed": args.seed, "blocks": args.blocks,
        "iters_per_block": args.iters_per_block, "total_iters": args.blocks * args.iters_per_block,
        "num_envs": args.num_envs, "ladder": rungs, "ep_len_cap": args.ep_len_cap,
        "crisp_threshold": args.crisp_threshold, "fuzzy_breaks": [flo, fmid, fhi],
        "warm_ckpt": warm, "final_ckpt": chain_ckpt, "smoke": args.smoke,
        "mu_path": [r["mu_trained"] for r in block_records],
    }
    with open(manifest_json, "w") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"\n[setB] CHAIN DONE arm={args.arm} seed={args.seed}")
    print(f"[setB] mu path: {manifest['mu_path']}")
    print(f"[setB] final ckpt: {chain_ckpt}")
    print(f"[setB] decisions: {decisions_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
