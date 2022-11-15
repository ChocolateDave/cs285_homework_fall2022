from __future__ import annotations

import numpy as np
from cs285.critics.base_critic import BaseCritic

# import pdb


class ArgMaxPolicy(object):

    def __init__(self,
                 critic: BaseCritic,
                 use_boltzmann: bool = False) -> None:
        self.critic = critic
        self.use_boltzmann = use_boltzmann

    def set_critic(self, critic: BaseCritic) -> None:
        self.critic = critic

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # MJ: changed the dimension check to a 3
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        # <DONE> return the action that maxinmizes the Q-value
        # at the current observation as the output
        q_values = self.critic.qa_values(observation)

        if self.use_boltzmann:
            distribution = np.exp(q_values) / np.sum(np.exp(q_values))
            action = self.sample_discrete(distribution)
        else:
            action = q_values.argmax(-1)

        return action[0]

    def sample_discrete(self, p: np.ndarray) -> np.ndarray:
        # https://stackoverflow.com/questions/40474436/how-to-apply-numpy-random-choice-to-a-matrix-of-probability-values-vectorized-s
        c = p.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        choices = (u < c).argmax(axis=1)
        return choices
