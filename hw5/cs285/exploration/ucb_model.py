from __future__ import annotations

from typing import Tuple

import cs285.infrastructure.pytorch_util as ptu
import gym
import numpy as np
from cs285.exploration.base_exploration_model import BaseExplorationModel
from torch import nn


class UCBModel(nn.Module, BaseExplorationModel):

    def __init__(self,
                 env: gym.Env,
                 cbe_coefficient: float = 1.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.cbe_coefficient = cbe_coefficient
        self.env = env
        self.counts = np.zeros(tuple(
            (self.env.observation_space.high -
             self.env.observation_space.low).astype(int)
        ))

    def _get_ob_bin(self, ob_no: np.ndarray) -> Tuple[np.ndarray, ...]:
        ob_no = ob_no * (
            self.env.observation_space.high - self.env.observation_space.low)
        assert len(self.env.observation_space.shape) == 1
        # Handle observations at the very high range of the observation space,
        # i.e. if the observation space is [0, 10] x [0, 8] then make sure
        # x coordinate is less than 10 and y coordinate is less than 8.
        for ax in range(self.env.observation_space.shape[0]):
            ob_no[ob_no[:, ax] >= self.env.observation_space.high[ax], ax] -= 1
        return tuple(np.floor(ob_no).astype(int).transpose())

    def forward(self, ob_no):
        return ptu.from_numpy(self.forward_np(ptu.to_numpy(ob_no)))

    def forward_np(self, ob_no):
        ob_counts = self.counts[self._get_ob_bin(ob_no)] + 1
        return self.cbe_coefficient / np.sqrt(ob_counts)

    def update(self, ob_no):
        self.counts[self._get_ob_bin(ob_no)] += 1
        return 0
