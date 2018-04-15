import argparse
import os.path as osp

import gym
import numpy as np
from mpi4py import MPI
from baselines import bench
from baselines import logger
from baselines.common import tf_util as U
from baselines.ppo1 import pposgd_simple, mlp_policy

import envs


def train(env_id):
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])

    env = gym.make(env_id)

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)

    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))

    def save_fn(local_vars, global_vars):
        if rank != 0:
            return
        vars_list = local_vars['var_list']
        vars_values = sess.run(vars_list)
        vars_dict = {var_name.name: np.array(var_value) for var_name, var_value in zip(vars_list, vars_values)}
        np.savez_compressed('./model.npz', **vars_dict)

    pposgd_simple.learn(
        env, policy_fn,
        max_episodes=1000000000,
        timesteps_per_actorbatch=200 * 300,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=1e-3, optim_batchsize=256,
        gamma=0.99, lam=0.95, schedule='constant',
        callback=save_fn
    )
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gym_id')
    args = parser.parse_args()

    train(args.gym_id)



if __name__ == '__main__':
    main()
