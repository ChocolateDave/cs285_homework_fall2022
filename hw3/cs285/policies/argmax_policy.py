import numpy as np
from ..critics.dqn_critic import DQNCritic


class ArgMaxPolicy(object):

    def __init__(self, critic: DQNCritic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) <= 3:
            obs = obs[None, ...]

        # Return the action that maxinmizes the Q-value
        # at the current observation as the output
        qa_val: np.ndarray = self.critic.qa_values(obs)
        action = qa_val.argmax(axis=1)

        return action.squeeze()
