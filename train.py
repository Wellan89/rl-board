# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
OpenAI gym execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import time
import uuid

import shutil
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

import envs
import csb
import export_to_codingame_submission


# python train.py csb-d0-v0 -a agents/trpo-v1.json -n networks/mlp-v1.json


def _restore(filename, agent):
    tensor_codes = {
        'dense0/W': 'trpo/actions-and-internals/layered-network/apply/dense0/apply/linear/apply/W',
        'dense0/b': 'trpo/actions-and-internals/layered-network/apply/dense0/apply/linear/apply/b',
        'dense1/W': 'trpo/actions-and-internals/layered-network/apply/dense1/apply/linear/apply/W',
        'dense1/b': 'trpo/actions-and-internals/layered-network/apply/dense1/apply/linear/apply/b',
        'alpha/W': 'trpo/actions-and-internals/beta/parameterize/alpha/apply/W',
        'alpha/b': 'trpo/actions-and-internals/beta/parameterize/alpha/apply/b',
        'beta/W': 'trpo/actions-and-internals/beta/parameterize/beta/apply/W',
        'beta/b': 'trpo/actions-and-internals/beta/parameterize/beta/apply/b',
    }

    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    with agent.model.graph.as_default():
        all_vars = tf.global_variables()
        for tensor_name in tensor_codes.values():
            matching_vars = [var for var in all_vars if var.op.name == tensor_name]
            assert len(matching_vars) == 1
            tensor_val = reader.get_tensor(tensor_name)
            matching_vars[0].load(tensor_val, agent.model.session)


class VersusOpponent:
    threshold_episodes_length = 1000

    def __init__(self, agent, reward_threshold):
        self.agent = agent
        self.reward_threshold = reward_threshold
        self.latest_rewards = []
        self.model = None
        self.reload()

    def episode_finished(self, latest_reward):
        self.latest_rewards.append(latest_reward)
        if len(self.latest_rewards) < self.threshold_episodes_length:
            return

        average_reward = sum(self.latest_rewards) / len(self.latest_rewards)
        if average_reward < self.reward_threshold:
            del self.latest_rewards[0]
        else:
            print('Reloading VersusOpponent: average reward is {:.2f} over the last {} episodes'.format(
                average_reward, self.threshold_episodes_length
            ))
            self.latest_rewards = []
            self.reload()

    def reload(self):
        save_dir = './versus_saved_models/{}/model'.format(uuid.uuid4())
        os.makedirs(os.path.dirname(save_dir))
        self.agent.save_model(save_dir, append_timestep=False)
        self.model = csb.Model.from_data(export_to_codingame_submission.read_weights(save_dir))
        shutil.rmtree(os.path.dirname(save_dir))

    def predict(self, state):
        return self.model.predict(state)


def _basename_no_ext(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-a', '--agent', help="Agent configuration file")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose actions deterministically")
    parser.add_argument('-l', '--load', default=None, help="Load agent from a previous checkpoint")
    parser.add_argument('--monitor', help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=1000, help="Save video every x steps (0 = disabled)")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()

    if not args.monitor:
        args.monitor = '{}_{}_{}'.format(args.gym_id, _basename_no_ext(args.agent), _basename_no_ext(args.network))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    if args.agent is not None:
        with open(args.agent, 'r') as fp:
            agent = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network is not None:
        with open(args.network, 'r') as fp:
            network = json.load(fp=fp)
    else:
        network = None
        logger.info("No network configuration provided.")

    environment = OpenAIGym(
        gym_id=args.gym_id,
        monitor='logs/{}/gym'.format(args.monitor),
        monitor_safe=args.monitor_safe,
        monitor_video=args.monitor_video
    )

    save_dir = 'logs/{}/checkpoints'.format(args.monitor)
    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network,
            saver=dict(directory=save_dir, seconds=600),
            summarizer=dict(directory='logs/{}/summaries'.format(args.monitor),
                            labels=[],
                            seconds=120)
        )
    )
    if args.load:
        # if not os.path.isdir(save_dir):
        #     raise OSError("Could not load agent from {}: No such directory.".format(save_dir))
        # agent.restore_model(save_dir)
        _restore(args.load, agent)

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent)

    versus_opponent = None
    versus_opponent_update_reward_threshold = getattr(environment.gym.unwrapped, 'versus_opponent_update_reward_threshold', 0.0)
    if versus_opponent_update_reward_threshold > 0.0:
        versus_opponent = VersusOpponent(agent, versus_opponent_update_reward_threshold)
        environment.gym.unwrapped.opp_solution_predict = versus_opponent.predict

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    if args.debug:
        report_episodes = 1
    else:
        report_episodes = 100

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            logger.info("Latest episode rewards: {}".format(', '.join(map('{:.2f}'.format, r.episode_rewards[-5:]))))
            logger.info("All time best: {:0.2f}".format(max(r.episode_rewards)))
            logger.info("1000 latest best: {:0.2f}".format(max(r.episode_rewards[-1000:])))
            logger.info("Average of last 500 rewards: {:0.2f}".format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {:0.2f}".format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
        if versus_opponent:
            versus_opponent.episode_finished(r.episode_rewards[-1])
        return True

    runner.run(
        timesteps=args.timesteps,
        episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished
    )
    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


if __name__ == '__main__':
    main()
