from __future__ import annotations

import math
from typing import Tuple

import cs285.infrastructure.pytorch_util as ptu
import gym
import numpy as np
import torch as th
from cs285.exploration.base_exploration_model import BaseExplorationModel
from torch import nn


class UCBModel(nn.Module, BaseExplorationModel):

    def __init__(self,
                 env: gym.Env,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.time_step: int = 0
        self.env = env
        self.state_counts = np.zeros(
            shape=list((env.observation_space.high -
                        env.action_space.low).astype(int)),
            dtype='int32'
        )

    def reset_parameters(self) -> None:
        self.time_step = 0

    def forward(self, ob_no: th.Tensor) -> th.Tensor:
        return ptu.from_numpy(self.forward_np(ptu.to_numpy(ob_no)))

    def foward_np(self, ob_no: np.ndarray) -> np.ndarray:
        n_ob_no = self.state_counts[self._get_ob_pos(ob_no)]
        return math.sqrt(2 * math.log(self.time_step)) / np.sqrt(n_ob_no)

    def update(self, ob_no: np.ndarray) -> np.ndarray:
        if isinstance(ob_no, th.Tensor):
            ob_no = ptu.to_numpy(ob_no)

        self.state_counts[self._get_ob_pos(ob_no)] += 1
        self.time_step += 1
        return np.array([0])

    def _get_ob_pos(self, ob_no:  np.ndarray) -> Tuple[np.ndarray, ...]:
        ob_no *= (self.env.observation_space.high -
                  self.env.observation_space.low)
        assert len(self.env.observation_space.shape) == 1, ValueError(
            'High dimensional observation space not supported!'
        )
        for dim in range(self.env.observation_space.shape[0]):
            ob_no[
                ob_no[:, dim] >= self.env.observation_space.high[dim],
                dim
            ] -= 1

        return list(np.floor(ob_no).astype(int).transpose())
