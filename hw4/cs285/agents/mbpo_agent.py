from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from cs285.agents.base_agent import BaseAgent
from cs285.agents.sac_agent import SACAgent
from cs285.agents.mb_agent import MBAgent
from cs285.infrastructure.utils import Path
from gym import Env


class MBPOAgent(BaseAgent):
    def __init__(self, env: Env, agent_params: Dict[str, Any]) -> None:
        super(MBPOAgent, self).__init__()
        self.mb_agent = MBAgent(env, agent_params)
        self.sac_agent = SACAgent(env, agent_params['sac_params'])
        self.env = env

        self.actor = self.sac_agent.actor

    def train(self, *args):
        return self.mb_agent.train(*args)

    def train_sac(self, *args):
        return self.sac_agent.train(*args)

    def collect_model_trajectory(self,
                                 rollout_length: int = 1
                                 ) -> List[Dict[str, np.ndarray]]:
        # Collect a trajectory of rollout_length from the learned
        # dynamics model. Start from a state sampled from the replay buffer.

        # sample 1 transition from self.mb_agent.replay_buffer
        ob, _, _, _, done = self.mb_agent.replay_buffer.sample_random_data(1)

        obs, acs, rews, next_obs, dones, img_obs = [], [], [], [], [], []
        for _ in range(rollout_length):
            # get the action from the policy
            ac = self.actor.get_action(ob)

            # determine the next observation by averaging
            # the prediction of all the dynamics models in the ensemble
            next_ob = np.mean([model.get_prediction(ob, ac)
                               for model in self.mb_agent.dyn_models], 0)

            # query the reward function to determine
            # the reward of this transition
            # HINT: use self.env.get_reward
            rew, _ = self.env.get_reward(ob, ac)

            obs.append(ob[0])
            acs.append(ac[0])
            rews.append(rew[0])
            next_obs.append(next_ob[0])
            dones.append(done[0])

            ob = next_ob
        return [Path(obs, img_obs, acs, rews, next_obs, dones)]

    def add_to_replay_buffer(self,
                             paths: List[Dict[str, np.ndarray]],
                             from_model: bool = False,
                             **kwargs) -> None:
        self.sac_agent.add_to_replay_buffer(paths)
        # only add rollouts from the real environment
        # to the model training buffer
        if not from_model:
            self.mb_agent.add_to_replay_buffer(paths, **kwargs)

    def sample(self, *args, **kwargs):
        return self.mb_agent.sample(*args, **kwargs)

    def sample_sac(self, *args, **kwargs):
        return self.sac_agent.sample(*args, **kwargs)
