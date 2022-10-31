from collections import OrderedDict

import gym
import torch as th
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from cs285.agents.base_agent import BaseAgent
from cs285.critics.sac_critic import SACCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.sac_policy import MLPPolicySAC
from numpy import ndarray


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
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self,
                      ob_no: ndarray,
                      ac_na: ndarray,
                      next_ob_no: ndarray,
                      re_n: ndarray,
                      terminal_n: ndarray) -> float:
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(re_n).unsqueeze(1)
        terminal_n = ptu.from_numpy(terminal_n).unsqueeze(1)

        with th.no_grad():
            dist = self.actor.forward(next_ob_no)
            next_action = dist.rsample()
            next_Qs = self.critic_target(next_ob_no, next_action)
            next_Q = th.min(*next_Qs)
            target_Q = reward_n + (1 - terminal_n) * self.gamma * next_Q
            next_log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q -= self.gamma * (1 - terminal_n) * \
                self.actor.alpha.detach() * next_log_prob

        critic_loss = 0
        # Get current Q values
        currentQs = self.critic(ob_no, ac_na)
        for current_Q in currentQs:
            critic_loss += self.critic.loss(current_Q, target_Q)

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        loss = OrderedDict()

        # Update critic function
        for _ in range(
                self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.update_critic(
                ob_no, ac_na, next_ob_no, re_n, terminal_n)
            loss['Critic_Loss'] = critic_loss
            if self.training_step % self.critic_target_update_frequency == 0:
                sac_utils.soft_update_params(
                    self.critic, self.critic_target, self.critic_tau)

        # Update policy function
        if self.training_step % self.actor_update_frequency == 0:
            for _ in range(
                    self.agent_params['num_actor_updates_per_agent_update']):
                actor_loss, alpha_loss, temperature = self.actor.update(
                    ob_no, self.critic)
                loss['Actor_Loss'] = actor_loss
                loss['Alpha_Loss'] = alpha_loss
                loss['Temperature'] = temperature

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
