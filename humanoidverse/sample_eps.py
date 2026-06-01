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

            # Capture training-time settings that eval_overrides would otherwise
            # change in ways that INVALIDATE the trained policy (vs. merely making
            # eval deployment-like). The soil eval_overrides (a) drop action_scale
            # 0.7->0.5 and stiffness ~20% (cripples the action->torque map), (b)
            # tighten termination (adds contact + low-height termination, stricter
            # tilt) so a fine gait is flagged as a fall, and (c) enable heavy action
            # smoothing the policy never trained with. Restore all three for a
            # train-matched eval unless explicitly opted out (eval_match_train=False)
            # for a deliberately deployment-like eval.
            train_match = None
            try:
                train_match = {
                    "action_scale": train_config.robot.control.action_scale,
                    "stiffness": OmegaConf.to_container(
                        train_config.robot.control.stiffness
                    ),
                    "damping": OmegaConf.to_container(
                        train_config.robot.control.damping
                    ),
                    "termination": OmegaConf.to_container(
                        train_config.env.config.termination
                    ),
                    "termination_scales": OmegaConf.to_container(
                        train_config.env.config.termination_scales
                    ),
                    "max_episode_length_s": train_config.env.config.max_episode_length_s,
                }
            except Exception:
                train_match = None

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)

            match_train = bool(
                OmegaConf.select(config, "eval_match_train", default=True)
            )
            if train_match is not None and match_train:
                config.robot.control.action_scale = train_match["action_scale"]
                config.robot.control.stiffness = train_match["stiffness"]
                config.robot.control.damping = train_match["damping"]
                config.env.config.termination = train_match["termination"]
                config.env.config.termination_scales = train_match[
                    "termination_scales"
                ]
                # Restore the training episode cap. eval_overrides sets this to
                # ~infinite (100000s), so a robust policy never terminates and the
                # eval hangs. The training cap (e.g. 20s) bounds the eval and makes
                # ep_len directly comparable to train-side numbers (cap = survived).
                config.env.config.max_episode_length_s = train_match[
                    "max_episode_length_s"
                ]
                # Policy never saw eval-time action smoothing during training.
                if "eval_action_smoothing" in config.env.config:
                    config.env.config.eval_action_smoothing = False
                logger.info(
                    "eval_match_train=True: restored training actuators "
                    f"(action_scale={train_match['action_scale']}), training "
                    "termination, and disabled eval action smoothing over "
                    "eval_overrides. Set eval_match_train=False for "
                    "deployment-like eval."
                )
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
    
    # Collect episode information
    ep_infos = []
    episodes_completed = 0
    
    # Get inference policy
    eval_policy = algo._get_inference_policy()
    obs_dict = env.reset_all()
    # ensure tensors on the same device as the model
    for k in list(obs_dict.keys()):
        if isinstance(obs_dict[k], torch.Tensor):
            obs_dict[k] = obs_dict[k].to(device)
    
    logger.info("Starting episode collection...")
    
    # Diagnostic: +eval_stochastic=True evaluates the SAMPLED (training-mode)
    # policy instead of the deterministic mean, to test whether a train/eval
    # gap is caused by exploration noise propping up a bad mean policy.
    eval_stochastic = bool(getattr(config, "eval_stochastic", False))
    if eval_stochastic:
        logger.info("[eval] STOCHASTIC mode: using actor.act() sampled actions")

    while episodes_completed < num_episodes:
        # Get actions from policy
        with torch.no_grad():
            if eval_stochastic:
                actions = algo.actor.act(obs_dict)  # sampled; ROA actor reads the dict
            else:
                actions = eval_policy(obs_dict["actor_obs"])  # deterministic mean
        
        # Take environment step
        actor_state = {"actions": actions}
        obs_dict, rewards, dones, infos = env.step(actor_state)

        # Mirror ppo.py logging: collect infos['episode'] (rew_* means) and
        # infos['to_log'] (env log_dict incl. dfh_*) for TB export.
        if tb_writer is not None and isinstance(infos, dict):
            ep_i = infos.get("episode")
            if ep_i:
                row = {k: _scalarize(v) for k, v in ep_i.items()}
                ep_info_accum.append({k: v for k, v in row.items() if v is not None})
            to_log = infos.get("to_log")
            if to_log:
                for k, v in to_log.items():
                    fv = _scalarize(v)
                    if fv is not None:
                        env_log_accum.setdefault(k, []).append(fv)

        # Accumulate step-wise metrics (IsaacSim focus; backend-agnostic when buffers exist)
        try:
            # Energy integral: sum(|tau * omega|) * dt
            power = torch.sum(torch.abs(env.torques) * torch.abs(env.simulator.dof_vel), dim=1)
            energy_accum += power * dt_control

            # Velocity error in base frame (xy)
            base_lv_world = env.simulator.robot_root_states[:, 7:10]
            base_quat = env.simulator.base_quat
            base_lv_local = quat_rotate_inverse(base_quat, base_lv_world)
            cmd_xy = env.commands[:, :2]
            vel_err = torch.linalg.norm(cmd_xy - base_lv_local[:, :2], dim=1)
            vel_err_sum += vel_err

            # Torso pitch (deg)
            torso_quat = env.simulator._rigid_body_rot[:, env.torso_index]
            rpy = get_euler_xyz_in_tensor(torso_quat)
            pitch_deg = rpy[:, 1] * (180.0 / math.pi)
            pitch_sum += pitch_deg
            pitch_sq_sum += pitch_deg * pitch_deg

            step_count += 1
        except Exception:
            pass
        # move obs back to device for next policy call
        for k in list(obs_dict.keys()):
            if isinstance(obs_dict[k], torch.Tensor):
                obs_dict[k] = obs_dict[k].to(device)
        
        # Check for completed episodes
        done_indices = torch.nonzero(dones, as_tuple=False).flatten()
        
        for env_idx in done_indices:
            env_idx_int = int(env_idx.item())

            # IMPORTANT: The environment resets done envs inside step().
            # Reading live buffers (episode_length_buf, slip_distance, start_pos)
            # here returns values for the NEW episode (often zero).
            # Instead, use the snapshot captured in _reset_tasks_callback(): env.episode_info
            episode_data = None
            if hasattr(env, 'episode_info'):
                episode_data = env.episode_info.get(f'env_{env_idx_int}', None)

            # Prepare advanced metrics for this completed episode
            sigma_pitch_deg = None
            cot = None
            vel_err_mean = None
            try:
                n = int(step_count[env_idx_int].item())
                if n > 0:
                    mean_pitch = float(pitch_sum[env_idx_int].item()) / n
                    mean_pitch_sq = float(pitch_sq_sum[env_idx_int].item()) / n
                    var_pitch = max(0.0, mean_pitch_sq - mean_pitch * mean_pitch)
                    sigma_pitch_deg = math.sqrt(var_pitch)
                    vel_err_mean = float(vel_err_sum[env_idx_int].item()) / n
            except Exception:
                pass

            if episode_data is not None:
                dist = float(episode_data.get('distance', 0.0))
                try:
                    if dist > 0.0:
                        cot = float(energy_accum[env_idx_int].item()) / (robot_mass * g * dist)
                except Exception:
                    cot = None
                episode_info = {
                    'distance': dist,
                    'slip_distance': float(episode_data.get('slip_distance', 0.0)),
                    'fell': bool(episode_data.get('fell', False)),
                    'episode_length': float(episode_data.get('episode_length', 0.0)),
                    'sigma_pitch_deg': sigma_pitch_deg,
                    'cot': cot,
                    'vel_err_mean': vel_err_mean,
                }
            else:
                # Fallback (best-effort) if env.episode_info is unavailable
                # Note: episode_length_buf is zeroed on reset; use last_episode_length_buf if present
                ep_len = 0.0
                if hasattr(env, 'last_episode_length_buf'):
                    try:
                        ep_len = float(env.last_episode_length_buf[env_idx_int].item())
                    except Exception:
                        ep_len = 0.0
                # Fall back for fell flag based on timeout buffer (set at termination time)
                fell_flag = False
                try:
                    fell_flag = (env.time_out_buf[env_idx_int].item() == False)
                except Exception:
                    fell_flag = False
                episode_info = {
                    'distance': 0.0,  # cannot reliably reconstruct after reset
                    'slip_distance': 0.0,  # cannot reliably reconstruct after reset
                    'fell': fell_flag,
                    'episode_length': ep_len,
                    'sigma_pitch_deg': sigma_pitch_deg,
                    'cot': None,
                    'vel_err_mean': vel_err_mean,
                }

            ep_infos.append(episode_info)
            episodes_completed += 1

            # Reset per-env accumulators after logging this episode
            energy_accum[env_idx_int] = 0.0
            vel_err_sum[env_idx_int] = 0.0
            step_count[env_idx_int] = 0
            pitch_sum[env_idx_int] = 0.0
            pitch_sq_sum[env_idx_int] = 0.0
            
            if episodes_completed % 10 == 0:
                logger.info(f"Completed {episodes_completed}/{num_episodes} episodes")
            
            if episodes_completed >= num_episodes:
                break
    
    logger.info(f"Evaluation completed. Collected {len(ep_infos)} episodes.")
    # Sanity warning for zero-length episodes
    try:
        zero_len_eps = sum(1 for info in ep_infos if float(info.get('episode_length', 0.0)) <= 0.0)
        if zero_len_eps > 0:
            logger.warning(f"Detected {zero_len_eps} episode(s) with zero/non-positive length; check reset timing.")
    except Exception:
        pass
    
    # Compute aggregate metrics over all evaluation episodes
    total_falls = sum(1 for info in ep_infos if info.get('fell'))
    total_dist = sum(info.get('distance', 0.0) for info in ep_infos)
    total_slip = sum(info.get('slip_distance', 0.0) for info in ep_infos)
    total_episode_length = sum(info.get('episode_length', 0) for info in ep_infos)
    # New metrics
    sigma_list = [info.get('sigma_pitch_deg') for info in ep_infos if info.get('sigma_pitch_deg') is not None]
    cot_list = [info.get('cot') for info in ep_infos if info.get('cot') is not None and math.isfinite(info.get('cot'))]
    vel_err_list = [info.get('vel_err_mean') for info in ep_infos if info.get('vel_err_mean') is not None]
    
    falls_per_100m = (total_falls / (total_dist/100.0)) if total_dist > 0 else 0.0
    slip_per_100m = (total_slip / (total_dist/100.0)) if total_dist > 0 else 0.0
    avg_episode_length = total_episode_length / len(ep_infos) if ep_infos else 0.0
    avg_distance = total_dist / len(ep_infos) if ep_infos else 0.0
    
    def _mean_and_ci(x):
        if not x:
            return (float('nan'), float('nan'))
        import numpy as np
        arr = np.array(x, dtype=float)
        m = float(arr.mean())
        s = float(arr.std(ddof=0))
        n = max(1, len(arr))
        ci = 1.96 * s / (n ** 0.5)
        return (m, ci)

    sigma_mean, sigma_ci = _mean_and_ci(sigma_list)
    cot_mean, cot_ci = _mean_and_ci(cot_list)
    vel_err_mean, vel_err_ci = _mean_and_ci(vel_err_list)

    # Print results to stdout
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes completed: {len(ep_infos)}")
    print(f"Total distance traveled: {total_dist:.2f} m")
    print(f"Total falls: {total_falls}")
    print(f"Average episode length: {avg_episode_length:.1f} steps")
    print(f"Average distance per episode: {avg_distance:.2f} m")
    print(f"Falls per 100m: {falls_per_100m:.2f}")
    print(f"Slip distance per 100m: {slip_per_100m:.2f} m")
    if sigma_list:
        print(f"Sigma pitch (deg): {sigma_mean:.2f} ± {sigma_ci:.2f}")
    if vel_err_list:
        print(f"Velocity error (m/s): {vel_err_mean:.3f} ± {vel_err_ci:.3f}")
    if cot_list:
        print(f"CoT (J/kg·m): {cot_mean:.2f} ± {cot_ci:.2f}")
    print("="*50)

    # Also log results via loguru so they appear in Hydra eval.log and any redirected stdout
    logger.info("==== EVALUATION RESULTS ====")
    logger.info(f"Episodes completed: {len(ep_infos)}")
    logger.info(f"Total distance traveled: {total_dist:.2f} m")
    logger.info(f"Total falls: {total_falls}")
    logger.info(f"Average episode length: {avg_episode_length:.1f} steps")
    logger.info(f"Average distance per episode: {avg_distance:.2f} m")
    logger.info(f"Falls per 100m: {falls_per_100m:.2f}")
    logger.info(f"Slip distance per 100m: {slip_per_100m:.2f} m")
    if sigma_list:
        logger.info(f"Sigma pitch (deg): {sigma_mean:.2f} ± {sigma_ci:.2f}")
    if vel_err_list:
        logger.info(f"Velocity error (m/s): {vel_err_mean:.3f} ± {vel_err_ci:.3f}")
    if cot_list:
        logger.info(f"CoT (J/kg·m): {cot_mean:.2f} ± {cot_ci:.2f}")

    # Write TB scalars (analyze_regret.py averages the last points per tag).
    if tb_writer is not None:
        if ep_info_accum:
            ep_keys = set().union(*(d.keys() for d in ep_info_accum))
            for key in ep_keys:
                vals = [d[key] for d in ep_info_accum if key in d]
                if vals:
                    tb_writer.add_scalar(f"Episode/{key}", sum(vals) / len(vals), 0)
        for k, vals in env_log_accum.items():
            if vals:
                tb_writer.add_scalar(f"Env/{k}", sum(vals) / len(vals), 0)
        # analyze_regret prefers Episode/mean_episode_length, else Train/.
        # avg_episode_length is in control steps, matching ppo.py's lenbuffer.
        tb_writer.add_scalar("Episode/mean_episode_length", float(avg_episode_length), 0)
        tb_writer.add_scalar("Train/mean_episode_length", float(avg_episode_length), 0)
        tb_writer.flush()
        tb_writer.close()
        logger.info(f"Wrote eval TensorBoard scalars to {eval_tb_dir}")

    # Flush stdout to ensure the summary is written when running under nohup/redirects
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
