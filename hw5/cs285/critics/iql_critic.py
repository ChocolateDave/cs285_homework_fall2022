from __future__ import annotations

from typing import Any, Dict

import cs285.infrastructure.pytorch_util as ptu
# import pdb
import numpy as np
import torch as th
import torch.optim as optim
from cs285.critics.base_critic import BaseCritic
from cs285.infrastructure.dqn_utils import OptimizerSpec
from torch import nn
from torch.nn import utils


class IQLCritic(BaseCritic):

    def __init__(self,
                 hparams: Dict[str, Any],
                 optimizer_spec: OptimizerSpec,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)

        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.mse_loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

        # TODO: define value function
        # HINT: see Q_net definition above and optimizer below
        self.v_net = network_initializer(self.ob_dim, 1)
        self.v_net.to(ptu.device)

        self.v_optimizer = self.optimizer_spec.constructor(
            self.v_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler_v = optim.lr_scheduler.LambdaLR(
            self.v_optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.iql_expectile = hparams['iql_expectile']

    def expectile_loss(self, diff: th.Tensor) -> th.Tensor:
        """
        Implement expectile loss on the difference between q and v
        """
        # TODO (Done): Compute expectile loss
        mask = diff < 0
        loss = th.abs(self.iql_expectile - mask.int()) * diff ** 2
        return loss

    def update_v(self,
                 ob_no: np.ndarray,
                 ac_na: np.ndarray) -> Dict[str, th.Tensor]:
        """
        Update value function using expectile loss
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(th.long)

        # TODO (Done): Compute value function loss
        q_vals = self.q_net(ob_no).gather(1, ac_na.unsqueeze(1)).squeeze(1)
        value_loss = self.expectile_loss(
            diff=q_vals.detach() - self.v_net(ob_no).squeeze(1)
        )
        value_loss = value_loss.mean()

        assert value_loss.shape == ()
        self.v_optimizer.zero_grad()
        value_loss.backward()
        utils.clip_grad_value_(self.v_net.parameters(),
                               self.grad_norm_clipping)
        self.v_optimizer.step()
        self.learning_rate_scheduler_v.step()

        return {'Training V Loss': ptu.to_numpy(value_loss)}

    def update_q(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
        Use target v network to train Q
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(th.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        # TODO (Done): Compute q function loss
        q_vals = self.q_net(ob_no).gather(1, ac_na.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(
            q_vals,
            reward_n + self.gamma * (1 - terminal_n) *
            self.v_net(next_ob_no).squeeze(1).detach()
        )

        assert loss.shape == ()
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(),
                               self.grad_norm_clipping)
        self.optimizer.step()

        self.learning_rate_scheduler.step()

        return {'Training Q Loss': ptu.to_numpy(loss)}

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
