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

    # Create and load algorithm
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)
    eval_policy = algo._get_inference_policy()

    import numpy as _np

    probe_out = getattr(config, "probe_out", "logs/S0Gate/probe/regime.npz")
    probe_steps = int(getattr(config, "probe_steps", 250))
    probe_regime = int(getattr(config, "probe_regime", 0))
    probe_regime_name = str(getattr(config, "probe_regime_name", f"regime{probe_regime}"))
    os.makedirs(os.path.dirname(probe_out), exist_ok=True)

    obs_dict = env.reset_all()
    for k in list(obs_dict.keys()):
        if isinstance(obs_dict[k], torch.Tensor):
            obs_dict[k] = obs_dict[k].to(device)

    n_envs = env.num_envs
    since_reset = torch.zeros(n_envs, dtype=torch.long, device=env.device)

    obs_buf = []      # [T, N, D]  exact policy input stream
    step_buf = []     # [T, N]     steps-since-episode-start (pre-fall vs late)
    done_buf = []     # [T, N]     episode boundary
    fell_buf = []     # [T, N]     1 if this done was a fall (not timeout)
    com_buf = []      # [T, N, 3]  privileged base-com bias (secondary target)
    push_buf = []     # [T, N, 2]  privileged push velocity (secondary target)

    logger.info(
        f"[probe] regime={probe_regime_name}({probe_regime}) steps={probe_steps} "
        f"n_envs={n_envs} -> {probe_out}"
    )

    for t in range(probe_steps):
        with torch.no_grad():
            actions = eval_policy(obs_dict["actor_obs"])
        ao = obs_dict["actor_obs"].detach().to(torch.float16).cpu().numpy()
        obs_dict, rewards, dones, infos = env.step({"actions": actions})
        for k in list(obs_dict.keys()):
            if isinstance(obs_dict[k], torch.Tensor):
                obs_dict[k] = obs_dict[k].to(device)

        since_reset += 1
        d = dones.detach().to(torch.bool).cpu()
        # fell = terminated but NOT a timeout (time_out_buf True => timeout)
        try:
            to = env.time_out_buf.detach().to(torch.bool).cpu()
            fell = (d & (~to))
        except Exception:
            fell = d.clone()

        obs_buf.append(ao)
        step_buf.append(since_reset.detach().cpu().numpy().astype(_np.int16))
        done_buf.append(d.numpy())
        fell_buf.append(fell.numpy())

        cb = getattr(env, "base_com_bias", None)
        if cb is not None:
            try:
                com_buf.append(cb.detach().float().cpu().numpy())
            except Exception:
                com_buf.append(_np.zeros((n_envs, 3), dtype=_np.float32))
        else:
            com_buf.append(_np.zeros((n_envs, 3), dtype=_np.float32))

        pv = getattr(env, "record_push_robot_vel_buf", None)
        if pv is not None:
            try:
                push_buf.append(pv.detach().float().cpu().numpy())
            except Exception:
                push_buf.append(_np.zeros((n_envs, 2), dtype=_np.float32))
        else:
            push_buf.append(_np.zeros((n_envs, 2), dtype=_np.float32))

        # reset step counter AFTER recording so frame t carries pre-reset age
        since_reset[dones.to(torch.bool)] = 0

        if (t + 1) % 50 == 0:
            logger.info(f"[probe] {probe_regime_name}: {t+1}/{probe_steps} steps")

    _np.savez_compressed(
        probe_out,
        obs=_np.stack(obs_buf, axis=0),
        ep_step=_np.stack(step_buf, axis=0),
        done=_np.stack(done_buf, axis=0),
        fell=_np.stack(fell_buf, axis=0),
        com_bias=_np.stack(com_buf, axis=0),
        push_vel=_np.stack(push_buf, axis=0),
        regime=_np.int64(probe_regime),
        regime_name=probe_regime_name,
    )
    logger.info(f"[probe] saved {probe_out}  obs_shape={_np.stack(obs_buf,0).shape}")

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    # Close simulator
    if 'simulation_app' in locals():
        simulation_app.close()

if __name__ == "__main__":
    main()
