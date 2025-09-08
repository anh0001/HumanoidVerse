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
    
    while episodes_completed < num_episodes:
        # Get actions from policy
        with torch.no_grad():
            actions = eval_policy(obs_dict["actor_obs"])  # policy expects actor_obs tensor
        
        # Take environment step
        actor_state = {"actions": actions}
        obs_dict, rewards, dones, infos = env.step(actor_state)
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

            if episode_data is not None:
                episode_info = {
                    'distance': float(episode_data.get('distance', 0.0)),
                    'slip_distance': float(episode_data.get('slip_distance', 0.0)),
                    'fell': bool(episode_data.get('fell', False)),
                    'episode_length': float(episode_data.get('episode_length', 0.0)),
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
                }

            ep_infos.append(episode_info)
            episodes_completed += 1
            
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
    
    falls_per_100m = (total_falls / (total_dist/100.0)) if total_dist > 0 else 0.0
    slip_per_100m = (total_slip / (total_dist/100.0)) if total_dist > 0 else 0.0
    avg_episode_length = total_episode_length / len(ep_infos) if ep_infos else 0.0
    avg_distance = total_dist / len(ep_infos) if ep_infos else 0.0
    
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
