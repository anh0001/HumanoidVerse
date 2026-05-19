"""PPO + Regularized Online Adaptation (ROA, Cheng et al. 2023).

Single-stage: a privileged teacher latent (from the already-asymmetric
`critic_obs`) and a proprio-history student latent (from `actor_obs`, which
carries `short_history` under the history obs config) are trained jointly with
the policy. The policy conditions on the student latent so PPO trains
policy+estimator end-to-end; a symmetric MSE regularizer ties student<->teacher.

Isolated subclass: the baseline `PPO` is untouched, so the no-ROA ablation
arms (A0/A3/A8) keep the exact original code path. Select with
`+algo=ppo_roa`. Inference uses only `actor_obs` so it stays drop-in
compatible with the existing eval / sample_eps path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.modules.ppo_modules import PPOActorROA, PPOCritic
from torch import optim


class PPOROA(PPO):
    def _setup_models_and_optimizer(self):
        roa_cfg = getattr(self.config, "roa", {}) or {}

        def _g(key, default):
            return roa_cfg[key] if key in roa_cfg else default

        self.roa_latent_dim = int(_g("latent_dim", 16))
        self.roa_coef = float(_g("roa_coef", 1.0))
        self.roa_beta = float(_g("beta", 0.5))  # weight on student->teacher term

        actor_obs_dim = int(self.algo_obs_dim_dict["actor_obs"])
        critic_obs_dim = int(self.algo_obs_dim_dict["critic_obs"])

        self.actor = PPOActorROA(
            actor_obs_dim=actor_obs_dim,
            critic_obs_dim=critic_obs_dim,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            latent_dim=self.roa_latent_dim,
            policy_hidden=list(_g("policy_hidden", [512, 256, 128])),
            est_hidden=list(_g("est_hidden", [256, 128])),
            priv_hidden=list(_g("priv_hidden", [128, 64])),
        ).to(self.device)

        self.critic = PPOCritic(
            self.algo_obs_dim_dict, self.config.module_dict.critic
        ).to(self.device)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.critic_learning_rate
        )
        logger.info(
            f"[ROA] latent_dim={self.roa_latent_dim} roa_coef={self.roa_coef} "
            f"beta={self.roa_beta} actor_obs={actor_obs_dim} critic_obs={critic_obs_dim}"
        )

    def _init_loss_dict_at_training_step(self):
        loss_dict = super()._init_loss_dict_at_training_step()
        loss_dict["ROA"] = 0
        return loss_dict

    def _actor_act_step(self, obs_dict):
        # ROA actor needs both actor_obs (student) and critic_obs (teacher).
        # obs_dict is the env obs dict in rollout and the minibatch dict in
        # _update_ppo; both carry these keys (registered in algo_obs_dim_dict).
        return self.actor.act(obs_dict)

    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict["actions"]
        target_values_batch = policy_state_dict["values"]
        advantages_batch = policy_state_dict["advantages"]
        returns_batch = policy_state_dict["returns"]
        old_actions_log_prob_batch = policy_state_dict["actions_log_prob"]
        old_mu_batch = policy_state_dict["action_mean"]
        old_sigma_batch = policy_state_dict["action_sigma"]

        self._actor_act_step(policy_state_dict)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy
        roa_teacher, roa_student = self.actor.roa_losses()
        roa_loss = roa_teacher + self.roa_beta * roa_student

        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)
                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)
                for pg in self.actor_optimizer.param_groups:
                    pg["lr"] = self.actor_learning_rate
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = self.critic_learning_rate

        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        entropy_loss = entropy_batch.mean()
        actor_loss = (
            surrogate_loss
            - self.entropy_coef * entropy_loss
            + self.roa_coef * roa_loss
        )
        critic_loss = self.value_loss_coef * value_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict["Value"] += value_loss.item()
        loss_dict["Surrogate"] += surrogate_loss.item()
        loss_dict["Entropy"] += entropy_loss.item()
        loss_dict["ROA"] += roa_loss.item()
        return loss_dict
