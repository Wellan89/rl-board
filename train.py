import argparse
import collections
import datetime
import os
import random
import threading

import gym
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from baselines import bench
from baselines import logger
from baselines.common import tf_util as U
from baselines.ppo1 import pposgd_simple, mlp_policy

import envs
import checkpoints_utils
import csb


def _load_vars_dict(local_vars):
    vars_list = local_vars['pi'].get_variables()
    vars_values = tf.get_default_session().run(vars_list)
    return {var.name: var_value for var, var_value in zip(vars_list, vars_values)}


class SaveCallback:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def _save(self, filename, vars_dict):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez_compressed(filename, **vars_dict)

    def __call__(self, local_vars, global_vars):
        vars_dict = _load_vars_dict(local_vars)
        self._save(os.path.join(self.log_dir, 'model.npz'), vars_dict)
        self._save(os.path.join(self.log_dir, 'checkpoints/model_{}.npz'.format(local_vars['iters_so_far'])), vars_dict)


class HardEnvCallback:
    def __init__(self, env, switch_iterations, linear_schedule):
        self.env = env.unwrapped
        self.switch_iterations = switch_iterations
        self.linear_schedule = linear_schedule
        self.printed = False

    def __call__(self, local_vars, global_vars):
        if self.linear_schedule:
            hard_env_weight = min(local_vars['iters_so_far'] / self.switch_iterations, 1.0)
        else:
            hard_env_weight = float(local_vars['iters_so_far'] >= self.switch_iterations)

        if not self.printed and hard_env_weight >= 1.0:
            print('HardEnvCallback: {} iterations done:'
                  ' done switching to hard environment version'.format(local_vars['iters_so_far']))
            self.printed = True

        self.env.set_hard_env_weight(hard_env_weight)


class VersusCallback:
    def __init__(self, env, start_iterations, threshold_iterations,
                 opp_update_reward_threshold, default_ai_weight):
        self.env = env.unwrapped
        self.env.enable_opponent(self.predict)
        self.start_iterations = start_iterations
        self.threshold_iterations = threshold_iterations
        self.opp_update_reward_threshold = opp_update_reward_threshold
        self.latest_rewards = collections.deque(maxlen=threshold_iterations)
        self.models = []
        self.current_model = None

        if default_ai_weight > 0:
            # This model is the default AI in the environment
            self.models += [None] * default_ai_weight

    def __call__(self, local_vars, global_vars):
        if not self.models:
            # Load the first model
            self.reload(local_vars)
            return

        if 'rews' not in local_vars or local_vars['iters_so_far'] < self.start_iterations:
            return

        self.latest_rewards.append(sum(local_vars['rews']) / len(local_vars['rews']))
        if len(self.latest_rewards) < self.threshold_iterations:
            return

        average_reward = sum(self.latest_rewards) / len(self.latest_rewards)
        if self.opp_update_reward_threshold is None or average_reward >= self.opp_update_reward_threshold:
            print('VersusCallback: Loading opponent {}: average reward is {:.2f} over the last {} iterations'.format(
                len(self.models), average_reward, len(self.latest_rewards)
            ))
            self.reload(local_vars)
            self.latest_rewards.clear()

    def reload(self, local_vars):
        self.models.append(csb.Model(_load_vars_dict(local_vars)))

    def predict(self, state, is_new_episode):
        if is_new_episode:
            self.current_model = random.choice(self.models)

        if self.current_model:
            return self.current_model.predict(state)
        else:
            return None


class ReloadCallback:
    def __init__(self, model_path):
        self.model_path = model_path

    def __call__(self, local_vars, global_vars):
        if not self.model_path:
            return

        print('Restoring from:', self.model_path)
        vars_dict = {var.name: var for var in local_vars['pi'].get_variables()}
        for var_name, var_value in checkpoints_utils.read_weights(self.model_path).items():
            vars_dict[var_name].load(var_value)
        self.model_path = None


# Two wrappers for the same environment needs to have a different name
class VideoMonitor(gym.wrappers.Monitor):
    pass


class VideoEpisodesMonitorCallback:
    def __init__(self):
        self._should_monitor = True

    def __call__(self, local_vars, global_vars):
        self._should_monitor = True

    def should_monitor(self, episode_idx):
        ret = self._should_monitor
        self._should_monitor = False
        return ret


class VideoMonitorCallback:
    def __init__(self, gym_id, log_dir, frequency_iters):
        self.gym_id = gym_id
        self.log_dir = log_dir
        self.frequency_iters = frequency_iters

    def _do_monitor(self, monitor_path, model):
        print('Recording run to:', monitor_path)
        env = VideoMonitor(gym.make(self.gym_id), monitor_path, video_callable=lambda _: True)
        state = env.reset()
        while True:
            action = model.predict(state)
            state, _, terminal, _ = env.step(action)
            if terminal:
                break
        env.close()
        print('Recorded run to:', monitor_path)

    def __call__(self, local_vars, global_vars):
        if local_vars['iters_so_far'] % self.frequency_iters != 0:
            return

        monitor_path = os.path.join(self.log_dir, 'video_monitor/{}'.format(local_vars['iters_so_far']))
        model = csb.Model(_load_vars_dict(local_vars))
        threading.Thread(target=self._do_monitor, args=(monitor_path, model)).start()


def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=80, num_hid_layers=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gym_id')
    parser.add_argument('-l', '--load')
    parser.add_argument('-e', '--episodes-per-batch', type=int, default=2000)
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    U.single_threaded_session().__enter__()

    log_dir = 'logs/{}_{}'.format(args.gym_id, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    logger.configure(dir=log_dir, format_strs=None if rank == 0 else [])

    env = bench.Monitor(gym.make(args.gym_id), os.path.join(logger.get_dir(), str(rank)))

    episodes_per_actorbatch = args.episodes_per_batch // size
    timesteps_per_actorbatch = episodes_per_actorbatch * 250

    callbacks = []
    if rank == 0:
        video_episodes_monitor_callback = VideoEpisodesMonitorCallback()
        callbacks.append(video_episodes_monitor_callback)
        monitor_path = os.path.join(log_dir, 'monitor')
        env = VideoMonitor(env, monitor_path, video_callable=video_episodes_monitor_callback.should_monitor)

    callbacks += [
        ReloadCallback(model_path=args.load),
        HardEnvCallback(env=env, switch_iterations=200, linear_schedule=True),
        VersusCallback(env=env, start_iterations=0, threshold_iterations=20,
                       opp_update_reward_threshold=None, default_ai_weight=3),
    ]
    if rank == 0:
        callbacks += [
            SaveCallback(log_dir=log_dir),
            # VideoMonitorCallback(gym_id=args.gym_id, log_dir=log_dir, frequency_iters=1),
        ]
    pposgd_simple.learn(
        env, policy_fn, max_iters=10000,
        timesteps_per_actorbatch=timesteps_per_actorbatch,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=1e-3, optim_batchsize=256,
        gamma=0.99, lam=0.95, schedule='constant',
        callback=lambda lv, gv: [cb(lv, gv) for cb in callbacks],
    )
    env.close()


if __name__ == '__main__':
    main()
