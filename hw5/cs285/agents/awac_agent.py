from __future__ import annotations

from typing import Any, Dict, Union

import cs285.infrastructure.pytorch_util as ptu
import gym
import numpy as np
import torch as th
from cs285.agents.dqn_agent import DQNAgent
from cs285.critics.dqn_critic import DQNCritic
from cs285.exploration.rnd_model import RNDModel
from cs285.infrastructure.dqn_utils import (MemoryOptimizedReplayBuffer,
                                            Schedule)
from cs285.infrastructure.utils import normalize
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.policies.MLP_policy import MLPPolicyAWAC


class AWACAgent(DQNAgent):
    def __init__(self,
                 env: gym.Env,
                 agent_params: Dict[str, Any],
                 normalize_rnd: bool = True,
                 rnd_gamma: float = 0.99) -> None:
        super(AWACAgent, self).__init__(env, agent_params)

        self.replay_buffer = MemoryOptimizedReplayBuffer(
            100000, 1, float_obs=True)
        self.num_exploration_steps = agent_params['num_exploration_steps']
        self.offline_exploitation = agent_params['offline_exploitation']

        self.exploitation_critic = DQNCritic(agent_params, self.optimizer_spec)
        self.exploration_critic = DQNCritic(agent_params, self.optimizer_spec)

        self.exploration_model = RNDModel(agent_params, self.optimizer_spec)
        self.explore_weight_schedule: Schedule = agent_params[
            'explore_weight_schedule']
        self.exploit_weight_schedule: Schedule = agent_params[
            'exploit_weight_schedule']

        self.actor = ArgMaxPolicy(self.exploitation_critic)
        self.eval_policy = self.awac_actor = MLPPolicyAWAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            self.agent_params['awac_lambda'],
        )

        self.exploit_rew_shift = agent_params['exploit_rew_shift']
        self.exploit_rew_scale = agent_params['exploit_rew_scale']
        self.eps = agent_params['eps']

        self.running_rnd_rew_std = 1
        self.normalize_rnd = normalize_rnd
        self.rnd_gamma = rnd_gamma

    def get_qvals(self,
                  critic: DQNCritic,
                  obs: Union[np.ndarray, th.Tensor],
                  action: Union[np.ndarray, th.Tensor]) -> th.Tensor:
        # TODO (DONE): get q-value for a given critic, obs, and action
        if isinstance(obs, np.ndarray):
            obs = ptu.from_numpy(obs)
        if isinstance(action, np.ndarray):
            action = ptu.from_numpy(action)

        qa_value: th.Tensor = critic.q_net_target.forward(obs)
        q_value = qa_value.gather(1, action.type(th.int64).unsqueeze(1))

        return q_value

    def estimate_advantage(self,
                           ob_no: np.ndarray,
                           ac_na: np.ndarray,
                           re_n: np.ndarray,
                           next_ob_no: np.ndarray,
                           terminal_n: np.ndarray,
                           n_actions: int = 10):
        # TODO (Done): Convert to torch tensors
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        re_n = ptu.from_numpy(re_n)
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)

        # Calculate and return the advantage (n sample estimate)
        # HINT: store computed values in the provided vals list.
        # You will use the average of this list for calculating the advantage.
        vals = []

        # TODO (Done): get action distribution for current obs,
        # you will use this for the value function estimate
        dist = self.eval_policy.forward(ob_no)

        # TODO (): Calculate Value Function Estimate
        # given current observation
        # HINT: You may find it helpful to utilze get_qvals defined above
        if self.agent_params['discrete']:
            for i in range(self.agent_params['ac_dim']):
                actions = th.ones_like(ac_na) * i
                vals.append(self.get_qvals(self.exploitation_critic,
                                           obs=ob_no, action=actions))
        else:
            for _ in range(n_actions):
                actions = dist.sample()
                vals.append(q_vals=self.get_qvals(self.exploitation_critic,
                                                  obs=ob_no, action=actions))
        v_pi = th.stack(vals).mean(dim=-1, keepdim=True)

        # TODO (Done): Calculate Q-Values
        q_vals = self.get_qvals(self.exploitation_critic, ob_no, ac_na)

        # TODO (Done): Calculate the Advantage using q_vals and v_pi
        return q_vals - v_pi

    def train(self,
              ob_no: np.ndarray,
              ac_na: np.ndarray,
              re_n: np.ndarray,
              next_ob_no: np.ndarray,
              terminal_n: np.ndarray) -> Dict[str, Any]:
        log = {}

        if self.t > self.num_exploration_steps:
            # TODO (Done): After exploration is over,
            # set the actor to optimize the extrinsic critic
            # HINT: Look at method ArgMaxPolicy.set_critic
            self.actor.set_critic(self.exploitation_critic)
            self.actor.use_boltzmann = False

        if (self.t > self.learning_starts
            and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)):
            # TODO (Done): Get Reward Weights
            # Get the current explore reward weight and exploit reward weight
            #       using the schedule's passed in (see __init__)
            # COMMENT: Until part 3, explore_weight = 1, and exploit_weight = 0
            explore_weight = self.explore_weight_schedule.value(self.t)
            exploit_weight = self.exploit_weight_schedule.value(self.t)

            # TODO (Done): Run Exploration Model
            # Evaluate the exploration model on s to get the exploration bonus
            # HINT: Normalize the exploration bonus,
            # as RND values vary highly in magnitude
            expl_bonus = self.exploration_model.forward_np(ob_no)
            if self.normalize_rnd:
                expl_bonus = normalize(data=expl_bonus,
                                       mean=expl_bonus.mean(),
                                       std=expl_bonus.std())

            # Reward Calculations
            # TODO (Done): Calculate mixed rewards,
            # which will be passed into the exploration critic
            # HINT: See doc for definition of mixed_reward
            mixed_reward = explore_weight * expl_bonus + exploit_weight * re_n

            # TODO (Done): Calculate the environment reward
            # HINT: For part 1, env_reward is just 're_n'
            # After this, env_reward is 're_n' shifted by
            # self.exploit_rew_shift, and scaled by self.exploit_rew_scale
            env_reward = self.exploit_rew_scale * \
                (re_n + self.exploit_rew_shift)

            # TODO (Done): Update Critics And Exploration Model
            # 1): Update the exploration model (based off s')
            # 2): Update the exploration critic (based off mixed_reward)
            # 3): Update the exploitation critic (based off env_reward)
            expl_model_loss = self.exploration_model.update(next_ob_no)
            expl_loss = self.exploration_critic.update(
                ob_no=ob_no, ac_na=ac_na, next_ob_no=next_ob_no,
                reward_n=mixed_reward, terminal_n=terminal_n
            )
            expt_loss = self.exploitation_critic.update(
                ob_no=ob_no, ac_na=ac_na, next_ob_no=next_ob_no,
                reward_n=env_reward, terminal_n=terminal_n
            )

            # TODO (Done): update actor
            # 1): Estimate the advantage
            # 2): Calculate the awac actor loss
            advantage = self.estimate_advantage(
                ob_no, ac_na, re_n, next_ob_no, terminal_n
            )
            actor_loss = self.eval_policy.update(ob_no, ac_na, advantage)

            # TODO (Done): Update Target Networks #
            if self.num_param_updates % self.target_update_freq == 0:
                #  Update the exploitation and exploration target networks
                self.exploration_critic.update_target_network()
                self.exploitation_critic.update_target_network()

            # Logging #
            log['Exploration Critic Loss'] = expl_loss['Training Loss']
            log['Exploitation Critic Loss'] = expt_loss['Training Loss']
            log['Exploration Model Loss'] = expl_model_loss

            # Uncomment these lines after completing awac
            log['Actor Loss'] = actor_loss

            self.num_param_updates += 1

        self.t += 1
        return log

    def step_env(self):
        """Step the env and store the transition

        At the end of this block of code, the simulator should have been
        advanced one step, and the replay buffer should contain one more
        transition. Note that self.last_obs must always point to the new
        latest observation.
        """
        if (not self.offline_exploitation) or \
                (self.t <= self.num_exploration_steps):
            self.replay_buffer_idx = self.replay_buffer.store_frame(
                self.last_obs)

        perform_random_action = np.random.random(
        ) < self.eps or self.t < self.learning_starts

        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            processed = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(processed)

        next_obs, reward, done, info = self.env.step(action)
        self.last_obs = next_obs.copy()

        if (not self.offline_exploitation) or \
                (self.t <= self.num_exploration_steps):
            self.replay_buffer.store_effect(
                self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.env.reset()
