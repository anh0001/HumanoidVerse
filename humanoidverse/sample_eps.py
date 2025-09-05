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
from utils.config_utils import *  # noqa: E402, F403

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger

@hydra.main(config_path="config", config_name="base_eval")
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
        parser.add_argument("--headless", type=bool, default=True, help="Run headless.")
        AppLauncher.add_app_launcher_args(parser)

        # Parse known arguments to get argparse params
        args_cli, hydra_args = parser.parse_known_args()

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
    env.set_is_evaluating(command=[0.0, 0.0, 0.0])  # Set to evaluation mode

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
    init_actions = torch.zeros(env.num_envs, env.dim_actions, device=device)
    
    logger.info("Starting episode collection...")
    
    while episodes_completed < num_episodes:
        # Get actions from policy
        with torch.no_grad():
            actions = eval_policy(obs_dict)
        
        # Take environment step
        actor_state = {"obs": obs_dict, "actions": actions}
        obs_dict, rewards, dones, infos = env.step(actor_state)
        
        # Check for completed episodes
        done_indices = torch.nonzero(dones, as_tuple=False).flatten()
        
        for env_idx in done_indices:
            env_idx_int = int(env_idx.item())
            
            # Calculate episode metrics
            start_pos = env.start_pos[env_idx_int]
            current_pos = env.simulator.robot_root_states[env_idx_int, :2]
            distance_traveled = torch.norm(current_pos - start_pos).item()
            
            slip_distance = env.slip_distance[env_idx_int].item()
            
            # Check if episode ended due to fall (termination)
            fell = env.time_out_buf[env_idx_int].item() == False  # Not a timeout = termination (fall)
            
            episode_info = {
                'distance': distance_traveled,
                'slip_distance': slip_distance, 
                'fell': fell,
                'episode_length': env.episode_length_buf[env_idx_int].item()
            }
            
            ep_infos.append(episode_info)
            episodes_completed += 1
            
            if episodes_completed % 10 == 0:
                logger.info(f"Completed {episodes_completed}/{num_episodes} episodes")
            
            if episodes_completed >= num_episodes:
                break
    
    logger.info(f"Evaluation completed. Collected {len(ep_infos)} episodes.")
    
    # Compute aggregate metrics over all evaluation episodes
    total_falls = sum(1 for info in ep_infos if info.get('fell'))
    total_dist = sum(info.get('distance', 0.0) for info in ep_infos)
    total_slip = sum(info.get('slip_distance', 0.0) for info in ep_infos)
    total_episode_length = sum(info.get('episode_length', 0) for info in ep_infos)
    
    falls_per_100m = (total_falls / (total_dist/100.0)) if total_dist > 0 else 0.0
    slip_per_100m = (total_slip / (total_dist/100.0)) if total_dist > 0 else 0.0
    avg_episode_length = total_episode_length / len(ep_infos) if ep_infos else 0.0
    avg_distance = total_dist / len(ep_infos) if ep_infos else 0.0
    
    # Print results
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
    
    # Close simulator
    if 'simulation_app' in locals():
        simulation_app.close()

if __name__ == "__main__":
    main()