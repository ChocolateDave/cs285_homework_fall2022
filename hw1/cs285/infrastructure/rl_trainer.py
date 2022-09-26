from collections import OrderedDict
# from os import path as osp
import numpy as np
import time
from typing import Any, Dict, Iterable, List

import pickle

import gym
import torch

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.logger import Logger
from cs285.infrastructure import utils
from cs285.policies import BasePolicy

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params: Dict[str, Any]) -> None:

        global MAX_VIDEO_LEN

        #############
        # INIT
        #############

        # Get params, create logger, create TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        # ENV
        #############

        # Make the gym environment
        if self.params['video_log_freq'] == -1:
            self.params['env_kwargs']['render_mode'] = None
        self.env = gym.make(self.params['env_name'],
                            **self.params['env_kwargs'])
        self.env.reset(seed=seed)

        # Maximum length for episodes
        if self.params['ep_len'] is None:
            self.params['ep_len'] = self.env.spec.max_episode_steps
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        if discrete:
            ac_dim = self.env.action_space.n
        else:
            ac_dim = self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        else:
            self.fps = self.env.env.metadata['render_fps']

        #############
        # AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """ Main training loop.

        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration to start relabel at
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************" % itr)

            # decide if videos should be rendered/logged at this iteration
            if (itr % self.params['video_log_freq'] == 0 and
                    self.params['video_log_freq'] != -1):
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr,
                initial_expertdata,
                collect_policy,
                self.params['batch_size']
            )  # HW1: implement this function below
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # Relabel the collected obs with actions
            # from a provided expert policy
            # HW1: implement this function below
            if relabel_with_expert and itr >= start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            # HW1: implement this function below
            training_logs = self.train_agent()

            # log/save
            if self.log_video or self.log_metrics:

                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(
                    itr, paths, eval_policy, train_video_paths, training_logs)

                if self.params['save_params']:
                    print('\nSaving agent params')
                    self.agent.save('{}/policy_itr_{}.pt'.format(
                        self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(self,
                                      itr: int,
                                      load_initial_expertdata: str,
                                      collect_policy: BasePolicy,
                                      batch_size: int) -> None:
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the total number of environment steps in paths
            train_video_paths: paths with videos for visualization
        """

        if itr == 0:
            with open(load_initial_expertdata, "rb") as p_file:
                loaded_paths = pickle.load(p_file)

            return loaded_paths, 0, None

        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(
            self.env, collect_policy, batch_size, self.params["ep_len"])

        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self.env,
                                                            collect_policy,
                                                            MAX_NVIDEO,
                                                            MAX_VIDEO_LEN,
                                                            True)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self) -> List[Dict[str, Any]]:
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = \
                self.agent.sample(self.params["train_batch_size"])

            # TODO use the sampled data to train an agent
            # HINT: use the agent's train function
            # HINT: keep the agent's training log for debugging
            train_log = self.agent.train(
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch
            )
            all_logs.append(train_log)

        return all_logs

    def do_relabel_with_expert(self, expert_policy, paths):
        print(
            "\nRelabelling collected observations "
            "with labels from an expert policy..."
        )

        for idx in range(len(paths)):
            paths[idx]["action"] = expert_policy.get_action(
                paths[idx]["observation"]
            )
        return paths

    ####################################
    ####################################

    def perform_logging(self, itr: int,
                        paths: Iterable[Dict[str, np.ndarray]],
                        eval_policy: BasePolicy,
                        train_video_paths: Iterable[Dict[str, np.ndarray]],
                        training_logs: Dict[str, Any]) -> None:

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
            self.env, policy=eval_policy,
            min_timesteps_per_batch=self.params['eval_batch_size'],
            max_path_length=self.params['ep_len']
        )

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths is not None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(
                self.env, policy=eval_policy, ntraj=MAX_NVIDEO,
                max_path_length=MAX_VIDEO_LEN, render=True
            )

            # save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths,
                                            step=itr,
                                            fps=self.fps,
                                            max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths,
                                            step=itr,
                                            fps=self.fps,
                                            max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum()
                            for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"])
                            for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            last_log = training_logs[-1]  # Only use the last log for now
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            # save returns
            # with open(osp.join(osp.dirname(osp.realpath(__file__)),
            #                    '../../data',
            #                    'eval_returns_' + str(time.time()) + '.pkl'),
            #           mode='wb') as fp:
            #     pickle.dump(np.array(eval_returns, dtype='float32'), fp,
            #                 protocol=3)

            self.logger.flush()
