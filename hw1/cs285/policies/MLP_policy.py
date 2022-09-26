import abc
import itertools
from typing import Any, Dict
from torch import nn
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim: int,
                 ob_dim: int,
                 n_layers: int,
                 size: int,
                 discrete: bool = False,
                 learning_rate: float = 1e-4,
                 training: bool = True,
                 nn_baseline: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            print("Logits Network:\n", self.logits_na)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.mean_net.to(ptu.device)
            print("Mean Net:\n", self.mean_net)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim,
                            dtype=torch.float32,
                            device=ptu.device)
            )
            self.logstd.to(ptu.device)
            print("Logstd Param:\n", self.logstd)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            # Batch input
            observation = obs
        else:
            # Extend batch dimension
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        with torch.no_grad():
            dist = self(observation)
            actions = dist.sample().cpu().numpy()

        return actions

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self,
                observation: torch.FloatTensor) -> distributions.Distribution:
        if self.logits_na:
            # act_hidden = F.softmax(self.logits_na(observation))
            act_hidden = self.logits_na(observation)
            dist = distributions.Categorical(prob=act_hidden)
        else:
            # Continuous action space
            act_mean = self.mean_net(observation)
            act_std = torch.ones_like(act_mean) * self.logstd.exp()
            dist = distributions.Normal(act_mean, act_std)

        return dist


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ) -> Dict[str, Any]:

        self.optimizer.zero_grad()
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        dist = self(observations)
        act_pred = dist.rsample()
        loss = self.loss(act_pred, actions)
        loss.backward()
        self.optimizer.step()
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
