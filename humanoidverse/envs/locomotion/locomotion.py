from time import time
from warnings import WarningMessage
import numpy as np
import os
import math

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from humanoidverse.envs.env_utils.general import class_to_dict
from humanoidverse.utils.spatial_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
# from humanoidverse.envs.env_utils.command_generator import CommandGenerator


class LeggedRobotLocomotion(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.init_done = True
        # import ipdb; ipdb.set_trace()
        if config.robot.motion.get("hips_link", None):
            self.hips_dof_id = [self.simulator._body_list.index(link) - 1 for link in config.robot.motion.hips_link] # Yuanhang: -1 for the base link (pelvis)
        
        # Initialize tracking for episode metrics (distance, slip)
        self.start_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.slip_distance = torch.zeros(self.num_envs, device=self.device)
        self.last_foot_pos = [[None, None] for _ in range(self.num_envs)]  # last contact position for each foot
        # Slip contact detection thresholds (force-based is robust on soil)
        self.slip_contact_force_threshold = 1.0  # normal force threshold (N) to consider foot in contact
        self.slip_contact_height_threshold = 0.03  # fallback z-height threshold (m)

    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.command_ranges = self.config.locomotion_command_ranges

    def _setup_simulator_control(self):
        self.simulator.commands = self.commands

    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading,
            compute measured terrain heights and randomly push robots
        """
        # 
        super()._update_tasks_callback()

        # commands
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
            # Only compute automatic yaw-rate command during training, not eval
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading),
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1]
            )

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set tiny commands to zero (lower threshold to encourage motion)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.05).unsqueeze(1)


    def _reset_tasks_callback(self, env_ids):
        # Collect episode info before reset
        if len(env_ids) > 0:
            if not hasattr(self, 'episode_info'):
                self.episode_info = {}
            
            for env_id in env_ids:
                env_id_int = int(env_id.item())
                
                # Calculate episode metrics
                if hasattr(self, 'start_pos'):
                    current_pos = self.simulator.robot_root_states[env_id_int, :2]
                    distance_traveled = torch.norm(current_pos - self.start_pos[env_id_int]).item()
                    slip_distance = self.slip_distance[env_id_int].item()
                    fell = not self.time_out_buf[env_id_int].item()  # Not timeout = fell
                    # Use last_episode_length_buf captured before zeroing in reset
                    if hasattr(self, 'last_episode_length_buf'):
                        episode_length = self.last_episode_length_buf[env_id_int].item()
                    else:
                        episode_length = self.episode_length_buf[env_id_int].item()
                    
                    # Store in episode info
                    self.episode_info[f'env_{env_id_int}'] = {
                        'distance': distance_traveled,
                        'slip_distance': slip_distance,
                        'fell': fell,
                        'episode_length': episode_length
                    }
        
        super()._reset_tasks_callback(env_ids)
        if not self.is_evaluating:
            self._resample_commands(env_ids)
        
        # On episode reset, record start position and reset slip metrics
        self.start_pos[env_ids] = self.simulator.robot_root_states[env_ids, :2]  # starting base (x,y) for each env
        self.slip_distance[env_ids] = 0.0
        for env_id in env_ids:
            env_id_int = int(env_id.item())
            self.last_foot_pos[env_id_int] = [None, None]

    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        self.commands = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        if command is None:
            self._resample_commands(torch.arange(self.num_envs, device=self.device))
        else:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands

    ########################### TRACKING REWARDS ###########################

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.config.rewards.reward_tracking_sigma.ang_vel)

    ########################### PENALTY REWARDS ###########################

    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_penalty_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_penalty_ang_vel_xy_torso(self):
        # Penalize xy axes base angular velocity

        torso_ang_vel = quat_rotate_inverse(self.simulator._rigid_body_rot[:, self.torso_index], self.simulator._rigid_body_ang_vel[:, self.torso_index])
        return torch.sum(torch.square(torso_ang_vel[:, :2]), dim=1)
    

    def _reward_penalty_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) -  self.config.rewards.locomotion_max_contact_force).clip(min=0.), dim=1)

    ########################### FEET REWARDS ###########################

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    # ---------------- Idea 2: periodic-clock + bilateral-symmetry ----------------
    # Siekmann-style swing/stance phase clock. Soft weights (see reward yaml) so
    # the clock cannot override terrain adaptation (reviewer risk #4). Phase is
    # derived directly from episode_length_buf so it is reset-safe automatically.
    def _gait_phase(self) -> torch.Tensor:
        period = float(getattr(self.config.rewards, "gait_period_s", 0.7))
        return (self.episode_length_buf.float() * self.dt / period) % 1.0

    @staticmethod
    def _expected_stance(phase: torch.Tensor, duty: float, ramp: float) -> torch.Tensor:
        """Smooth expected-contact indicator in [0,1]: ~1 during stance
        ([0, duty)), ~0 during swing ([duty, 1)), cosine-ramped over ``ramp``
        (cycle fraction) at the stance->swing transition."""
        t = torch.clamp((phase - (duty - ramp)) / ramp, 0.0, 1.0)  # 0 -> 1 across ramp
        ramp_val = 0.5 * (1.0 + torch.cos(torch.pi * t))           # 1 -> 0
        stance = torch.where(phase < (duty - ramp), torch.ones_like(phase), ramp_val)
        return torch.where(phase >= duty, torch.zeros_like(phase), stance)

    def _reward_gait_phase(self) -> torch.Tensor:
        """Reward feet matching a periodic swing/stance clock (left/right
        antiphase). Bounded ~[0,1]; zeroed at near-zero command."""
        duty = float(getattr(self.config.rewards, "gait_duty", 0.6))
        ramp = float(getattr(self.config.rewards, "gait_ramp", 0.1))
        phase = self._gait_phase()
        es_l = self._expected_stance(phase, duty, ramp)
        es_r = self._expected_stance((phase + 0.5) % 1.0, duty, ramp)
        contact = (self.simulator.contact_forces[:, self.feet_indices, 2] > 1.).float()
        c_l, c_r = contact[:, 0], contact[:, 1]
        match_l = es_l * c_l + (1.0 - es_l) * (1.0 - c_l)
        match_r = es_r * c_r + (1.0 - es_r) * (1.0 - c_r)
        rew = 0.5 * (match_l + match_r)
        rew *= (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        return rew

    def _reward_penalty_gait_asymmetry(self) -> torch.Tensor:
        """Bilateral-symmetry penalty: unequal left/right swing durations.
        No joint mirror map needed (robust). Soft weight; zeroed at zero command."""
        asym = torch.square(self.feet_air_time[:, 0] - self.feet_air_time[:, 1])
        asym *= (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        return asym
    
    def _reward_penalty_in_the_air(self):
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_foot_contact = contact_filt[:,0]
        second_foot_contact = contact_filt[:,1]
        reward = ~(first_foot_contact | second_foot_contact)
        return reward



    def _reward_penalty_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.simulator.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.simulator.contact_forces[:, self.feet_indices, 2]), dim=1)


    def _reward_penalty_sinkage_excess(self):
        """Quadratic penalty when local DFH sinkage at the feet exceeds sigma.

        Pulls per-foot plastic depth from the DFH adapter
        (``simulator._dfh.layer``) for the env_idx + cell at the foot XY.
        Returns zeros when DFH is disabled. ``terms.penalty_sinkage_excess.sigma_m``
        controls the soft excess threshold (default 0.04 m).
        """
        sim = self.simulator
        adapter = getattr(sim, "_dfh", None)
        if adapter is None:
            return torch.zeros(self.num_envs, device=self.device)
        try:
            layer = adapter.layer
            h_plastic = layer.h_plastic  # (N, H, W) signed metres
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)
        feet_xy = sim._rigid_body_pos[:, self.feet_indices, :2]  # (N, n_feet, 2)
        env_origins = getattr(sim, "env_origins", None)
        if env_origins is not None:
            origins_xy = env_origins[:, :2].unsqueeze(1)  # (N, 1, 2)
        else:
            origins_xy = torch.zeros_like(feet_xy[:, :1, :])
        scale = float(layer.cfg.horizontal_scale_m)
        gh, gw = layer.cfg.grid_h, layer.cfg.grid_w
        ext_x = gh * scale
        ext_y = gw * scale
        local = feet_xy - origins_xy
        local[..., 0] = local[..., 0] + 0.5 * ext_x
        local[..., 1] = local[..., 1] + 0.5 * ext_y
        cell_i = torch.clamp(torch.floor(local[..., 0] / scale).long(), 0, gh - 1)
        cell_j = torch.clamp(torch.floor(local[..., 1] / scale).long(), 0, gw - 1)
        env_grid = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(cell_i)
        depth = h_plastic[env_grid, cell_i, cell_j]  # signed metres (N, n_feet)
        sink = depth.clamp(max=0.0).abs()  # >=0 metres
        sigma = 0.04
        try:
            terms = getattr(self.config.rewards, "terms", None)
            if terms is not None:
                term_cfg = getattr(terms, "penalty_sinkage_excess", None)
                if term_cfg is not None:
                    sigma = float(getattr(term_cfg, "sigma_m", sigma))
        except Exception:
            pass
        excess = (sink - sigma).clamp(min=0.0)
        return torch.sum(excess * excess, dim=1)

    def _reward_penalty_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _reward_base_height(self):
        # Penalize base height away from target. Use a relative height computed
        # against the local ground (approximated by the lowest foot height).
        # Using absolute world-Z on height-field terrains (e.g., furrows) can
        # produce very large errors because the terrain origin varies across the
        # tile. The relative metric is robust to local undulations.
        base_height = self.simulator.robot_root_states[:, 2]
        feet_z = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        ground_z = torch.min(feet_z, dim=1).values
        rel_height = base_height - ground_z
        return torch.square(rel_height - self.config.rewards.desired_base_height)

    def _reward_penalty_hip_pos(self):
        # Penalize the hip joints (only roll and yaw)
        hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
        hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
        return torch.sum(torch.square(hip_pos), dim=1)

    def _reward_feet_heading_alignment(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]

        forward_left_feet = quat_apply(left_quat, self.forward_vec)
        heading_left_feet = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
        forward_right_feet = quat_apply(right_quat, self.forward_vec)
        heading_right_feet = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])


        root_forward = quat_apply(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(wrap_to_pi(heading_left_feet - heading_root))
        heading_diff_right = torch.abs(wrap_to_pi(heading_right_feet - heading_root))
        
        return heading_diff_left + heading_diff_right
    
    def _reward_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _reward_penalty_feet_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    

    def _reward_penalty_feet_height(self):
        # Penalize base height away from target
        feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2]
        dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 
    
    def _reward_penalty_close_feet_xy(self):
        # returns 1 if two feet are too close
        left_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[0], :2]
        right_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[1], :2]
        feet_distance_xy = torch.norm(left_foot_xy - right_foot_xy, dim=1)
        return (feet_distance_xy < self.config.rewards.close_feet_threshold) * 1.0
    

    def _reward_penalty_close_knees_xy(self):
        # returns 1 if two knees are too close
        left_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[0], :2]
        right_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[1], :2]
        self.knee_distance_xy = torch.norm(left_knee_xy - right_knee_xy, dim=1)
        return (self.knee_distance_xy < self.config.rewards.close_knees_threshold)* 1.0
    

    def _reward_upperbody_joint_angle_freeze(self):
        # returns keep the upper body joint angles close to the default
        assert self.config.robot.has_upper_body_dof
        deviation = torch.abs(self.simulator.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:,self.upper_dof_indices])
        return torch.sum(deviation, dim=1)
    
    def _post_physics_step(self):
        super()._post_physics_step()
        
        # Update slip-distance tracker for feet in contact with ground
        left_pos = self.simulator._rigid_body_pos[:, self.feet_indices[0]]   # (num_envs, 3)
        right_pos = self.simulator._rigid_body_pos[:, self.feet_indices[1]]  # (num_envs, 3)
        # Prefer contact-force gating; OR with z-height as conservative fallback
        foot_normal_forces = self.simulator.contact_forces[:, self.feet_indices, 2]  # (num_envs, 2)
        left_contact_force = foot_normal_forces[:, 0] > self.slip_contact_force_threshold
        right_contact_force = foot_normal_forces[:, 1] > self.slip_contact_force_threshold
        left_contact_height = left_pos[:, 2] < self.slip_contact_height_threshold
        right_contact_height = right_pos[:, 2] < self.slip_contact_height_threshold
        left_contact = torch.logical_or(left_contact_force, left_contact_height)
        right_contact = torch.logical_or(right_contact_force, right_contact_height)
        
        for env in range(self.num_envs):
            # Left foot slip
            if left_contact[env]:
                if self.last_foot_pos[env][0] is not None:
                    dx = float(left_pos[env, 0] - self.last_foot_pos[env][0][0])
                    dy = float(left_pos[env, 1] - self.last_foot_pos[env][0][1])
                    self.slip_distance[env] += math.sqrt(dx*dx + dy*dy)
                self.last_foot_pos[env][0] = (float(left_pos[env, 0]), float(left_pos[env, 1]))
            else:
                self.last_foot_pos[env][0] = None
            
            # Right foot slip
            if right_contact[env]:
                if self.last_foot_pos[env][1] is not None:
                    dx = float(right_pos[env, 0] - self.last_foot_pos[env][1][0])
                    dy = float(right_pos[env, 1] - self.last_foot_pos[env][1][1])
                    self.slip_distance[env] += math.sqrt(dx*dx + dy*dy)
                self.last_foot_pos[env][1] = (float(right_pos[env, 0]), float(right_pos[env, 1]))
            else:
                self.last_foot_pos[env][1] = None
    
    ######################### Observations #########################
    def _get_obs_command_lin_vel(self):
        return self.commands[:, :2]
    
    def _get_obs_command_ang_vel(self):
        return self.commands[:, 2:3]
