import argparse
import datetime
import os

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
    vars_list = local_vars['var_list']
    vars_values = tf.get_default_session().run(vars_list)
    return {var.name: var_value for var, var_value in zip(vars_list, vars_values)}


class SaveCallback:
    def __init__(self, rank, log_dir):
        self.rank = rank
        self.log_dir = log_dir

    def __call__(self, local_vars, global_vars):
        if self.rank == 0:
            filename = os.path.join(self.log_dir, 'model.npz')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, **_load_vars_dict(local_vars))


class HardEnvCallback:
    def __init__(self, env, switch_iterations):
        self.env = env.unwrapped
        self.switch_iterations = switch_iterations

    def __call__(self, local_vars, global_vars):
        if self.switch_iterations is None or local_vars['iters_so_far'] < self.switch_iterations:
            return

        print('HardEnvCallback: {} iterations done:'
              ' switching to hard environment version'.format(local_vars['iters_so_far']))
        self.switch_iterations = None
        try:
            self.env.switch_to_hard_env()
        except Exception as e:
            print('HardEnvCallback: Error: could not switch to hard environment:', e)


class VersusCallback:
    def __init__(self, env, start_iterations, threshold_iterations, opp_update_reward_threshold):
        self.env = env.unwrapped
        self.start_iterations = start_iterations
        self.threshold_iterations = threshold_iterations
        self.opp_update_reward_threshold = opp_update_reward_threshold
        self.latest_rewards = []
        self.model = None

    def __call__(self, local_vars, global_vars):
        if local_vars['iters_so_far'] < self.start_iterations:
            return

        if not self.model:
            print('Loading opponent')
            self.reload(local_vars)
            return

        self.latest_rewards.append(sum(local_vars['rews']) / len(local_vars['rews']))
        if len(self.latest_rewards) < self.threshold_iterations:
            return

        average_reward = sum(self.latest_rewards) / len(self.latest_rewards)
        if average_reward < self.opp_update_reward_threshold:
            del self.latest_rewards[0]
        else:
            print('VersusCallback: Reloading opponent: average reward is {:.2f} over the last {} iterations'.format(
                average_reward, len(self.latest_rewards)
            ))
            self.reload(local_vars)

    def reload(self, local_vars):
        self.latest_rewards = []
        self.model = csb.Model(_load_vars_dict(local_vars))
        self.env.enable_opponent(self.predict)

    def predict(self, state):
        return self.model.predict(state)


class ReloadCallback:
    def __init__(self, model_path):
        self.model_path = model_path

    def __call__(self, local_vars, global_vars):
        if not self.model_path:
            return

        print('Restoring from:', self.model_path)
        vars_dict = {var.name: var for var in local_vars['var_list']}
        for var_name, var_value in checkpoints_utils.read_weights(self.model_path).items():
            vars_dict[var_name].assign(var_value)
        self.model_path = None


def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gym_id')
    parser.add_argument('-l', '--load')
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    U.single_threaded_session().__enter__()

    log_dir = 'logs/{}_{}'.format(args.gym_id, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    logger.configure(dir=log_dir, format_strs=None if rank == 0 else [])

    env = bench.Monitor(gym.make(args.gym_id), os.path.join(logger.get_dir(), str(rank)))

    callbacks = [
        ReloadCallback(model_path=args.load),
        SaveCallback(rank=rank, log_dir=log_dir),
        HardEnvCallback(env=env, switch_iterations=70),
        VersusCallback(env=env, start_iterations=100, threshold_iterations=5, opp_update_reward_threshold=2.0),
    ]
    pposgd_simple.learn(
        env, policy_fn, max_iters=1000000,
        timesteps_per_actorbatch=2000 * 300 // size,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=1e-3, optim_batchsize=256,
        gamma=0.99, lam=0.95, schedule='constant',
        callback=lambda lv, gv: [cb(lv, gv) for cb in callbacks],
    )
    env.close()


if __name__ == '__main__':
    main()
