from collections import OrderedDict

import cs285.infrastructure.pytorch_util as ptu
import gym
import torch
from cs285.critics.sac_critic import SACCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.sac_utils import soft_update_params
from cs285.infrastructure.utils import copy
from cs285.policies.sac_policy import MLPPolicySAC
from torch.nn import functional as F

from .base_agent import BaseAgent


class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params[
            'actor_update_frequency'
        ]
        self.critic_target_update_frequency = self.agent_params[
            'critic_target_update_frequency'
        ]

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # 1. Compute the target Q value.
        # HINT: You need to use the entropy term (alpha)
        policy = self.actor.forward(next_ob_no)
        next_ac_na = policy.rsample()
        q_1_prime, q_2_prime = self.critic_target.forward(
            obs=next_ob_no,
            action=next_ac_na
        )
        min_target_qs = torch.min(q_1_prime, q_2_prime)
        entropy = policy.log_prob(next_ac_na).sum(-1, keepdim=True)
        target: torch.Tensor = \
            re_n + self.gamma * (1 - terminal_n) * (
                min_target_qs -
                self.actor.alpha.detach() * entropy
            )
        target = target.detach()

        # 2. Get current Q estimates and calculate critic loss
        qs = self.critic.forward(ob_no, ac_na)
        critic_loss = F.mse_loss(qs[0], target) + F.mse_loss(qs[1], target)

        # 3. Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        for step in range(
            self.agent_params['num_critic_updates_per_agent_update']
        ):
            critic_loss = self.update_critic(
                ob_no=ptu.from_numpy(ob_no),
                ac_na=ptu.from_numpy(ac_na),
                re_n=ptu.from_numpy(re_n),
                next_ob_no=ptu.from_numpy(next_ob_no),
                terminal_n=ptu.from_numpy(terminal_n)
            )

        # 2. Softly update the target every critic_target_update_frequency
        # (HINT: look at sac_utils)
            if step % self.critic_target_update_frequency == 0:
                soft_update_params(
                    net=self.critic,
                    target_net=self.critic_target,
                    tau=self.critic_tau
                )

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        for _ in range(
            self.agent_params['num_actor_updates_per_agent_update']
        ):
            actor_loss, alpha_loss, temperature = self.actor.update(
                obs=ptu.from_numpy(ob_no),
                critic=self.critic
            )

        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = temperature

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
