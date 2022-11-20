from __future__ import annotations

from typing import Any, Dict, Tuple

import cs285.infrastructure.pytorch_util as ptu
import torch as th
import torch.optim as optim
from cs285.critics.base_critic import BaseCritic
from cs285.infrastructure.dqn_utils import OptimizerSpec
from torch import nn

# import pdb


class CQLCritic(BaseCritic):

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
        self.optimizer: optim.Optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.cql_alpha = hparams['cql_alpha']

    def dqn_loss(self,
                 ob_no: th.Tensor,
                 ac_na: th.Tensor,
                 next_ob_no: th.Tensor,
                 reward_n: th.Tensor,
                 terminal_n: th.Tensor
                 ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # TODO (Done): Implement TD-error DQN loss with double q-learning
        if reward_n.dim() == 1:
            reward_n = reward_n.view(-1, 1)
        if terminal_n.dim() == 1:
            terminal_n = terminal_n.view(-1, 1)

        qa_t_values = self.q_net.forward(ob_no)
        q_t_values = th.gather(qa_t_values, 1, ac_na.unsqueeze(1))
        qa_t_values_tar = self.q_net_target.forward(next_ob_no)

        if self.double_q:
            next_ac_na = self.q_net.forward(next_ob_no).argmax(1)  # double q
            q_t_values_tar = th.gather(
                qa_t_values_tar, 1, next_ac_na.unsqueeze(1))
        else:
            q_t_values_tar = qa_t_values.max(1)

        bellman_tar = reward_n + self.gamma * (1 - terminal_n) * q_t_values_tar
        loss = self.loss.forward(q_t_values, bellman_tar.detach())

        return loss, qa_t_values, q_t_values

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n) -> None:
        """Update the parameters of the critic.

        let `sum_of_path_lengths` be the sum of the lengths of the paths
        sampled from Agent.sample_trajectories.
        let `num_paths` be the number of paths sampled from
        Agent.sample_trajectories

        Args:
            ob_no: shape: (sum_of_path_lengths, ob_dim)
            next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation
            after taking one step forward
            reward_n: length: sum_of_path_lengths. Each element in `reward_n`
            is a scalar containing the reward for each timestep
            terminal_n: length: sum_of_path_lengths. Each element in
            `terminal_n` is either 1 if the episode ended at that timestep
            or 0 if the episode did not end
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(th.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        # Compute the DQN Loss
        loss, qa_t_values, q_t_values = self.dqn_loss(
            ob_no, ac_na, next_ob_no, reward_n, terminal_n
        )

        # CQL Implementation
        # TODO (Done): Implement CQL as described in the pdf and paper
        # Hint: After calculating cql_loss, augment the loss appropriately
        self.optimizer.zero_grad()
        q_t_logsumexp = qa_t_values.logsumexp(dim=1)
        cql_loss = (q_t_logsumexp - q_t_values.detach()).mean()
        loss = loss + self.cql_alpha * cql_loss
        loss.backward()
        th.nn.utils.clip_grad.clip_grad_norm_(
            self.q_net.parameters(), self.grad_norm_clipping
        )
        self.optimizer.step()

        info = {'Training Loss': ptu.to_numpy(loss)}

        # Uncomment these lines after implementing CQL
        info['CQL Loss'] = ptu.to_numpy(cql_loss)
        info['Data q-values'] = ptu.to_numpy(q_t_values).mean()
        info['OOD q-values'] = ptu.to_numpy(q_t_logsumexp).mean()

        self.learning_rate_scheduler.step()

        return info

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
