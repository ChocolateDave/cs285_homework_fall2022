from typing import Dict, Iterable, List, Tuple

import numpy as np
# import time
from cs285.policies import BasePolicy
from gym import Env
from numpy import ndarray

# Type alias
_Path = Dict[str, ndarray]

############################################
############################################

MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]
MJ_ENV_KWARGS = {name: {"render_mode": "rgb_array"} for name in MJ_ENV_NAMES}
MJ_ENV_KWARGS["Ant-v4"]["use_contact_forces"] = True


def sample_trajectory(env: Env,
                      policy: BasePolicy,
                      max_path_length: int,
                      render: bool = False) -> Dict[str, ndarray]:

    # initialize env for the beginning of a new rollout
    ob = env.reset()  # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if hasattr(env, 'sim'):
                image_obs.append(env.sim.render(camera_name='track',
                                                height=500,
                                                width=500)[::-1])
            else:
                image_obs.append(env.render())

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        try:
            ob, rew, done, _ = env.step(ac)
        except Exception as e:
            raise RuntimeError(f"{e}, ac={ac}")

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        rollout_done = done or steps == max_path_length
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(
    env: Env,
    policy: BasePolicy,
    min_timesteps_per_batch: int,
    max_path_length: int,
    render: bool = False
) -> Tuple[Iterable[_Path], int]:
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.
    """
    timesteps_this_batch = 0
    paths: List[_Path] = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path: _Path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(env: Env,
                          policy: BasePolicy,
                          ntraj: int,
                          max_path_length: int,
                          render: bool = False) -> List[_Path]:
    """Collect ntraj rollouts.

    Args:
        env: An MuJoCo `BaseEnv` object as the simulation environment.
        policy: A `BasePolicy` object as the action policy.
        ntraj: An integer total number of trajectories to sample.
        max_path_length: An integer maximum path timesteps.
        render: A boolean flag if render the simulation.

    Returns:
        A list of `_Path` dictionaries.
    """
    paths = []
    cntr: int = 0
    while cntr < ntraj:
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
        cntr += 1

    return paths

############################################
############################################


def Path(obs: Iterable[ndarray],
         image_obs: Iterable[ndarray],
         acs: Iterable[ndarray],
         rewards: Iterable[ndarray],
         next_obs: Iterable[ndarray],
         terminals: Iterable[ndarray]) -> Dict[str, ndarray]:
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation": np.array(obs, dtype=np.float32),
            "image_obs": np.array(image_obs, dtype=np.uint8),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(
    paths: Iterable[_Path],
    concat_rew: bool = True
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Take a list of rollout dictionaries and return separate arrays.

    Each array is a concatenation of that array from across the rollouts.

    Args:
        paths: An iterable of `_Path` objects.
        concat_rew: A boolean flag if concatenate rewards.

    Returns:
        A tuple of `ndarray` objects each with a shape of `[n, *]`, where
        `n` is the number of paths.
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"]
                                        for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])

    return observations, actions, rewards, next_observations, terminals

############################################
############################################


def get_pathlength(path):
    return len(path["reward"])
