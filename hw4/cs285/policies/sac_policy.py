from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np
import torch as th
from cs285.critics.sac_critic import SACCritic
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.MLP_policy import MLPPolicy
from torch import nn, optim
from torch.distributions import Distribution


class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20, 2],
                 action_range=[-1, 1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers,
                                           size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = th.tensor(
            np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = th.optim.Adam(
            [self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        dist = self.forward(observation)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        action = ptu.to_numpy(action)

        return action

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a th.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: th.FloatTensor) -> Distribution:
        mean = self.mean_net.forward(observation)
        log_std = th.tanh(self.logstd)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = th.clip(log_std, log_std_min, log_std_max)
        std = log_std.exp()
        dist = sac_utils.SquashedNormal(mean, std)

        return dist

    def update(self,
               obs: np.ndarray,
               critic: SACCritic) -> Tuple:
        obs = ptu.from_numpy(obs)

        # Update policy function
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Qs = critic.forward(obs, action)
        actor_Q = th.min(*actor_Qs)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Update temperature
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha
