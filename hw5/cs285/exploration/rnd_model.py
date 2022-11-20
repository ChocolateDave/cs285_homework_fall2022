from __future__ import annotations

from typing import Any, Dict, Union

import cs285.infrastructure.pytorch_util as ptu
import numpy as np
import torch as th
import torch.optim as optim
from cs285.exploration.base_exploration_model import BaseExplorationModel
from cs285.infrastructure.dqn_utils import OptimizerSpec
from torch import nn


def init_method_1(model: th.nn.Module):
    model.weight.data.uniform_()
    model.bias.data.uniform_()


def init_method_2(model: th.nn.Module):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self,
                 hparams: Dict[str, Any],
                 optimizer_spec: OptimizerSpec,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # TODO (Done): Create two neural networks:
        # 1) f, the random function we are trying to learn.
        # 2) f_hat, the function we are using to learn f.
        # HINT: to prevent zero prediction error right from the beginning,
        # initialize the two networks using two different schemes.
        self.f = ptu.build_mlp(input_size=self.ob_dim,
                               output_size=self.output_size,
                               n_layers=self.n_layers,
                               size=self.size,
                               init_method=init_method_1).to(ptu.device)
        self.f_hat = ptu.build_mlp(input_size=self.ob_dim,
                                   output_size=self.output_size,
                                   n_layers=self.n_layers,
                                   size=self.size,
                                   init_method=init_method_2).to(ptu.device)
        self.optimizer: optim.Optimizer = self.optimizer_spec.constructor(
            self.f.parameters(), **self.optimizer_spec.optim_kwargs
        )

    def forward(self, ob_no: th.Tensor) -> th.Tensor:
        # TODO (Done): Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        tar = self.f.forward(ob_no).detach()
        pred = self.f_hat.forward(ob_no)

        return th.norm(pred - tar, dim=1)

    def forward_np(self, ob_no: np.ndarray) -> np.ndarray:
        ob_no: th.Tensor = ptu.from_numpy(ob_no)
        error = self.forward(ob_no)

        return ptu.to_numpy(error)

    def update(self, ob_no: Union[np.ndarray, th.Tensor]) -> np.ndarray:
        # TODO (Done): Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        if isinstance(ob_no, np.ndarray):
            ob_no = ptu.from_numpy(ob_no)
        self.optimizer.zero_grad()
        loss = self.forward(ob_no).mean()  # mean across the batch
        loss.backward()
        self.optimizer.step()

        return ptu.to_numpy(loss)
