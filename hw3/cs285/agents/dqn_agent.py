from __future__ import annotations

from typing import Any, Dict, List, Tuple

import gym
import numpy as np
from cs285.critics.dqn_critic import DQNCritic
from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from cs285.policies.argmax_policy import ArgMaxPolicy


class DQNAgent(object):
    def __init__(self, env: gym.Env, agent_params: Dict[str, Any]) -> None:

        self.env: gym.Env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        # import ipdb; ipdb.set_trace()
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'],
            agent_params['frame_history_len'],
            lander=lander
        )
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths: List[Dict[str, np.ndarray]]) -> None:
        pass

    def step_env(self) -> None:
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain
            one more transition.
            Note that `self.last_obs` must always point to
            the new latest observation.
        """

        # Store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is the
        # `MemoryOptimizedReplayBuffer` in dqn_utils.py
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        eps = self.exploration.value(self.t)

        # Use epsilon-greedy exploration when selecting action
        perform_random_action = (np.random.random() < eps or
                                 self.t < self.learning_starts)
        if perform_random_action:
            # HINT: take random action (can sample from self.env.action_space)
            # with probability eps (see np.random.random())
            # OR if your current step number (see self.t) is less
            # than self.learning_starts
            action = self.env.action_space.sample()
        else:
            # HINT: Your actor will take in multiple previous observations
            # ("frames") in order to deal with the partial observability of
            # the environment. Get the most recent `frame_history_len`
            # observations using functionality from the replay buffer,
            # and then use those observations as input to your actor.
            action = self.actor.get_action(
                self.replay_buffer.encode_recent_observation()
            )

        # Take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to
        # the newest/latest observation
        # HINT2: remember the following useful function that
        # you've seen before: obs, reward, done, info = env.step(action)
        self.last_obs, reward, done, _ = self.env.step(action)

        # Store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in
        # is self.replay_buffer_idx from above
        self.replay_buffer.store_effect(
            idx=self.replay_buffer_idx,
            action=action,
            reward=reward,
            done=done
        )

        # If taking this step resulted in done,
        # reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()

    def sample(self, batch_size: int) -> Tuple[List, List, List, List, List]:
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [], [], [], [], []

    def train(self,
              ob_no: np.ndarray,
              ac_na: np.ndarray,
              re_n: np.ndarray,
              next_ob_no: np.ndarray,
              terminal_n: np.ndarray) -> Dict[str, Any]:
        log = {}
        if (
            self.t > self.learning_starts
            and self.t % self.learning_freq == 0
            and self.replay_buffer.can_sample(self.batch_size)
        ):

            # Fill in the call to the update function using
            # the appropriate tensors
            log = self.critic.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n
            )

            # Update the target network periodically
            # HINT: your critic already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
