import os
import sys
import math
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger
# Use the 2-arg helpers used throughout the envs to avoid signature mismatch
from humanoidverse.utils.torch_utils import quat_rotate_inverse
from humanoidverse.utils.spatial_utils.rotations import get_euler_xyz_in_tensor

@hydra.main(config_path="config", config_name="base_eval", version_base="1.1")
def main(override_config: OmegaConf):
    # logging to hydra log file
    hydra_log_path = os.path.join(
        HydraConfig.get().runtime.output_dir, "eval.log"
    )
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    # Load training config from checkpoint
    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        config = override_config
            
    # Setup simulator
    simulator_type = config.simulator['_target_'].split('.')[-1]
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate an RL agent and collect episode metrics.")
        parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to simulate.")
        parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to run.")
        parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
        # Do NOT add --headless here; AppLauncher adds it and will conflict if duplicated.
        AppLauncher.add_app_launcher_args(parser)

        # Parse known arguments to get argparse params
        args_cli, hydra_args = parser.parse_known_args()

        # Sync Hydra overrides into AppLauncher args (mirrors train_agent.py)
        if hasattr(config, 'num_envs'):
            args_cli.num_envs = config.num_envs
        if hasattr(config, 'seed'):
            args_cli.seed = config.seed
        # env spacing lives under env.config in our configs
        if hasattr(config, 'env') and hasattr(config.env, 'config') and hasattr(config.env.config, 'env_spacing'):
            args_cli.env_spacing = config.env.config.env_spacing
        if hasattr(config, 'output_dir'):
            args_cli.output_dir = config.output_dir
        if hasattr(config, 'headless'):
            args_cli.headless = config.headless

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        sys.argv = [sys.argv[0]] + hydra_args
    if simulator_type == 'IsaacGym':
        import isaacgym
        
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    import torch
    
    pre_process_config(config)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Get evaluation parameters
    num_episodes = getattr(config, 'num_episodes', 100)

    logger.info(f"Running evaluation for {num_episodes} episodes")

    # Optional TensorBoard logging so cross-eval runs feed analyze_regret.py,
    # which globs events.out.tfevents.* and averages the last points of the
    # Episode/* and Env/dfh_* tags -- the SAME tags ppo.py emits at train time
    # (Episode/<k> from infos['episode'], Env/<k> from infos['to_log']).
    eval_tb_dir = getattr(config, 'eval_tb_dir', None)
    tb_writer = None
    ep_info_accum: list[dict[str, float]] = []  # mirrors ppo.py ep_infos averaging
    env_log_accum: dict[str, list[float]] = {}  # mirrors ppo.py Env/ tensors
    if eval_tb_dir:
        from torch.utils.tensorboard import SummaryWriter as _TBWriter
        os.makedirs(eval_tb_dir, exist_ok=True)
        tb_writer = _TBWriter(log_dir=eval_tb_dir, flush_secs=10)
        logger.info(f"TensorBoard eval logging -> {eval_tb_dir}")

    def _scalarize(v: object) -> float | None:
        try:
            return float(v.float().mean().item()) if hasattr(v, "float") else float(v)
        except Exception:
            return None
    
    # Create environment
    env = instantiate(config.env, device=device)
    # Log effective timing for sanity checks
    try:
        sim_dt = getattr(env, 'sim_dt', None)
        dt = getattr(env, 'dt', None)
        max_eplen_s = getattr(env, 'max_episode_length_s', None)
        max_eplen = getattr(env, 'max_episode_length', None)
        ctrl_dec = None
        try:
            ctrl_dec = config.simulator.config.sim.control_decimation
        except Exception:
            pass
        logger.info(
            f"Eval timing: sim_dt={sim_dt:.5f}s, control_decimation={ctrl_dec}, dt={dt:.5f}s, "
            f"max_episode_length_s={max_eplen_s}, max_episode_length={int(max_eplen) if max_eplen is not None else 'NA'}"
        )
    except Exception:
        pass

    # Optional: constant eval command via CLI/Hydra override
    # Usage example:
    #   +eval_command=[0.2,0.0,0.0]
    def _parse_eval_command(cmd):
        try:
            import collections.abc
            if cmd is None:
                return None
            # Already a list/tuple
            if isinstance(cmd, collections.abc.Sequence) and not isinstance(cmd, str):
                vals = [float(x) for x in cmd]
                return vals
            # String form: "[a,b,c]" or "a,b,c"
            if isinstance(cmd, str):
                s = cmd.strip()
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                parts = [p.strip() for p in s.split(",") if p.strip()]
                if not parts:
                    return None
                return [float(p) for p in parts]
        except Exception:
            pass
        return None

    eval_command = None
    try:
        # Allow passing a custom constant command from CLI (Hydra adds unknown keys)
        eval_command = _parse_eval_command(getattr(config, "eval_command", None))
    except Exception:
        eval_command = None

    if eval_command is None:
        eval_command = [0.0, 0.0, 0.0]

    env.set_is_evaluating(command=eval_command)  # Set to evaluation mode

    # ---- Per-env accumulators for advanced metrics ----
    num_envs = env.num_envs
    dt_control = float(getattr(env, 'dt', 0.02))
    g = 9.81
    # Determine robot mass (prefer simulator-provided; allow +robot_mass_kg override; else default)
    try:
        robot_mass = float(getattr(env.simulator, 'robot_mass', None))
    except Exception:
        robot_mass = None
    try:
        mass_override = getattr(config, 'robot_mass_kg', None)
        if mass_override is not None:
            robot_mass = float(mass_override)
    except Exception:
        pass
    if robot_mass is None:
        robot_mass = 60.0

    energy_accum = torch.zeros(num_envs, device=env.device)
    vel_err_sum = torch.zeros(num_envs, device=env.device)
    step_count = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    pitch_sum = torch.zeros(num_envs, device=env.device)
    pitch_sq_sum = torch.zeros(num_envs, device=env.device)

    # ---- Spawn probe: no policy. Reset, log spawn geometry, zero-action steps. ----
    import numpy as _np

    probe_tag = str(getattr(config, "probe_tag", "probe"))
    probe_steps = int(getattr(config, "probe_steps", 5))
    out_path = str(getattr(config, "probe_out", f"logs/SpawnProbe/{probe_tag}.txt"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _stats(t):
        t = t.detach().float().cpu()
        return f"mean={t.mean().item():.4f} min={t.min().item():.4f} max={t.max().item():.4f}"

    lines = [f"=== SPAWN PROBE: {probe_tag} ==="]

    obs_dict = env.reset_all()
    sim = env.simulator
    feet = env.feet_indices
    term_idx = env.termination_contact_indices

    def snapshot(label):
        base_z = sim.robot_root_states[:, 2]
        foot_z = sim._rigid_body_pos[:, feet, 2]            # [N, n_feet]
        env_org_z = sim.env_origins[:, 2]
        term_cf = torch.norm(sim.contact_forces[:, term_idx, :], dim=-1)  # [N, n_term]
        foot_cf = torch.norm(sim.contact_forces[:, feet, :], dim=-1)
        term_hit = (term_cf > 1.0).any(dim=1).float()
        lines.append(f"-- {label} --")
        lines.append(f"  base_z          : {_stats(base_z)}")
        lines.append(f"  env_origin_z    : {_stats(env_org_z)}")
        lines.append(f"  foot_z (min)    : {_stats(foot_z.min(dim=1).values)}")
        lines.append(f"  base_z - env_org_z (height above origin): {_stats(base_z - env_org_z)}")
        lines.append(f"  foot_z - env_org_z (foot clearance vs origin): {_stats(foot_z.min(dim=1).values - env_org_z)}")
        lines.append(f"  termination-body contact force |F|: {_stats(term_cf.amax(dim=1))}")
        lines.append(f"  foot contact force |F|           : {_stats(foot_cf.amax(dim=1))}")
        lines.append(f"  envs with non-foot contact >1N (term-trigger): "
                     f"{int(term_hit.sum().item())}/{env.num_envs}")
        try:
            lines.append(f"  reset_buf set: {int(env.reset_buf.sum().item())}/{env.num_envs}")
        except Exception:
            pass

    snapshot("AT RESET (step 0, before any action)")

    zero_act = torch.zeros(env.num_envs, env.config.robot.actions_dim, device=device)
    for s in range(1, probe_steps + 1):
        obs_dict, rewards, dones, infos = env.step({"actions": zero_act})
        n_done = int(dones.sum().item())
        lines.append(f"  [zero-action step {s}] envs terminated this step: {n_done}/{env.num_envs}")
        if s == 1:
            snapshot("AFTER 1 ZERO-ACTION STEP")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines:
        logger.info(ln)
    logger.info(f"[spawn_probe] wrote {out_path}")

    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass

    # Close simulator
    if 'simulation_app' in locals():
        simulation_app.close()

if __name__ == "__main__":
    main()
