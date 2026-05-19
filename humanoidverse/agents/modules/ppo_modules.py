from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal

from .modules import BaseModule

class PPOActor(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                num_actions,
                init_noise_std):
        super(PPOActor, self).__init__()

        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        # Clamp learned std to a sane range to avoid explosive actions
        # that we observed on compliant terrains (furrows/soil).
        sigma = torch.clamp(self.std, 0.05, 2.0)
        self.distribution = Normal(mean, mean * 0.0 + sigma)

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs):
        actions_mean = self.actor(actor_obs)
        return actions_mean
    
    def to_cpu(self):
        self.actor = deepcopy(self.actor).to('cpu')
        self.std.to('cpu')



class _MLP(nn.Module):
    """Small MLP helper for the ROA encoders/policy head."""

    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, act: str = "ELU"):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        a = getattr(nn, act)
        for h in hidden:
            layers += [nn.Linear(d, h), a()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOActorROA(nn.Module):
    """Regularized Online Adaptation actor (Cheng et al. 2023), single stage.

    - priv_encoder: critic_obs (privileged/asymmetric, e.g. base_lin_vel that
      the actor never sees) -> teacher latent z.
    - estimator:    actor_obs (proprio + short_history) -> student latent zhat.
    - policy:       [actor_obs, zhat] -> action mean.

    The policy acts on zhat (so PPO trains policy+estimator end-to-end); the
    ROA regularizer (added in PPOROA._update_ppo) ties zhat<->z. Inference uses
    only actor_obs -> zhat, so it is drop-in compatible with the existing eval
    path (`act_inference(actor_obs_tensor)`).
    """

    def __init__(self, actor_obs_dim: int, critic_obs_dim: int, num_actions: int,
                 init_noise_std: float, latent_dim: int = 16,
                 policy_hidden: list[int] | None = None,
                 est_hidden: list[int] | None = None,
                 priv_hidden: list[int] | None = None):
        super().__init__()
        policy_hidden = policy_hidden or [512, 256, 128]
        est_hidden = est_hidden or [256, 128]
        priv_hidden = priv_hidden or [128, 64]
        self.latent_dim = latent_dim
        self.priv_encoder = _MLP(critic_obs_dim, priv_hidden, latent_dim)
        self.estimator = _MLP(actor_obs_dim, est_hidden, latent_dim)
        self.policy = _MLP(actor_obs_dim + latent_dim, policy_hidden, num_actions)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self._z = None       # teacher latent (set in act)
        self._zhat = None    # student latent (set in act)
        Normal.set_default_validate_args = False

    # --- API mirroring PPOActor ---
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _mean_from(self, actor_obs: torch.Tensor, zhat: torch.Tensor) -> torch.Tensor:
        return self.policy(torch.cat([actor_obs, zhat], dim=-1))

    def update_distribution(self, obs_dict: dict):
        actor_obs = obs_dict["actor_obs"]
        self._zhat = self.estimator(actor_obs)
        self._z = self.priv_encoder(obs_dict["critic_obs"])
        mean = self._mean_from(actor_obs, self._zhat)
        sigma = torch.clamp(self.std, 0.05, 2.0)
        self.distribution = Normal(mean, mean * 0.0 + sigma)

    def act(self, obs_dict: dict, **kwargs):
        self.update_distribution(obs_dict)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs: torch.Tensor):
        # deploy path: proprio history only, no privileged obs
        zhat = self.estimator(actor_obs)
        return self._mean_from(actor_obs, zhat)

    def roa_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(teacher-toward-student, student-toward-teacher) MSE terms."""
        z, zhat = self._z, self._zhat
        return (
            torch.mean((z - zhat.detach()) ** 2),
            torch.mean((zhat - z.detach()) ** 2),
        )


class PPOCritic(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict):
        super(PPOCritic, self).__init__()

        self.critic_module = BaseModule(obs_dim_dict, module_config_dict)

    @property
    def critic(self):
        return self.critic_module
    
    def reset(self, dones=None):
        pass
    
    def evaluate(self, critic_obs, **kwargs):
        value = self.critic(critic_obs)
        return value
