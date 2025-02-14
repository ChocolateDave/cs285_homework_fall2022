from __future__ import annotations

from typing import List, Optional

import numpy as np
from cs285.models.base_model import BaseModel
from cs285.policies.base_policy import BasePolicy
from gym import Env


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env: Env,
                 ac_dim: int,
                 dyn_models: List[BaseModel],
                 horizon: int,
                 N: int,
                 sample_strategy: str = 'random',
                 cem_iterations: int = 4,
                 cem_num_elites: int = 5,
                 cem_alpha: int = 1,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models: List[BaseModel] = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, \
            f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                  f"num_elites={self.cem_num_elites}, "
                  f"iterations={self.cem_iterations}")

    def sample_action_sequences(self,
                                num_sequences: int,
                                horizon: int,
                                obs: Optional[np.ndarray] = None
                                ) -> np.ndarray:
        if self.sample_strategy == 'random' \
                or (self.sample_strategy == 'cem' and obs is None):
            # Uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            random_action_sequences = np.random.uniform(
                low=self.low,
                high=self.high,
                size=(num_sequences, horizon, self.ac_dim)
            )
            return random_action_sequences

        elif self.sample_strategy == 'cem':
            # Implement action selection using CEM.
            # Begin with randomly selected actions,
            # then refine the sampling distribution
            # iteratively as described in Section 3.3,
            # "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf
            cem_action = np.random.uniform(
                low=self.low,
                high=self.high,
                size=(num_sequences, horizon, self.ac_dim)
            )
            cem_loc = self.cem_alpha * cem_action.mean(0)
            cem_scale = self.cem_alpha * cem_action.std(0)
            for i in range(self.cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #   (Hint: remember that for the first iteration,
                #   we instead sample uniformly at random just like
                #   we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute
                #       rewards for our candidate sequences in order
                #       to rank them?)
                # - Update the elite mean and variance
                cem_action = np.random.normal(
                    loc=cem_loc,
                    scale=cem_scale,
                    size=(num_sequences, horizon, self.ac_dim)
                )
                cem_action = np.clip(cem_action, self.low, self.high)
                elite_idcs = np.argsort(self.evaluate_candidate_sequences(
                    cem_action, obs))[-self.cem_num_elites:]
                cem_action = cem_action[elite_idcs]
                cem_loc = self.cem_alpha * cem_action.mean(0) + \
                    (1 - self.cem_alpha) * cem_loc
                cem_scale = self.cem_alpha * cem_action.std(0) + \
                    (1 - self.cem_alpha) * cem_scale

            # Set `cem_action` to the appropriate action
            # chosen by CEM
            cem_action = cem_loc

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self,
                                     candidate_action_sequences: np.ndarray,
                                     obs: np.ndarray) -> np.ndarray:
        # For each model in ensemble, compute the predicted sum of
        # rewards for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
        result = []
        for model in self.dyn_models:
            result.append(self.calculate_sum_of_rewards(
                obs, candidate_action_sequences, model))

        return np.mean(result, axis=0)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider;
            # return the first action
            return candidate_action_sequences[0][0][None]

        else:
            predicted_rewards = self.evaluate_candidate_sequences(
                candidate_action_sequences, obs)

            # pick the action sequence and return
            # the 1st element of that sequence
            best_action_sequence = candidate_action_sequences[
                np.argmax(predicted_rewards)]
            action_to_take = best_action_sequence[0]
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self,
                                 obs: np.ndarray,
                                 candidate_action_sequences: np.ndarray,
                                 model: BaseModel) -> np.ndarray:
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the
        candidate actions sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.
        sum_of_rewards = np.zeros(shape=(self.N,))
        assert candidate_action_sequences.shape[:2] == (self.N, self.horizon)
        obs = np.tile(obs, (self.N, 1))
        for t in range(self.horizon):
            acs = candidate_action_sequences[:, t, :]
            rews, _ = self.env.get_reward(obs, acs)
            assert rews.shape == (self.N, )
            sum_of_rewards += rews
            obs = model.get_prediction(obs, acs, self.data_statistics)

        return sum_of_rewards
