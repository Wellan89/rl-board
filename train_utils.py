import json
import logging
import os
import shutil
import time
import uuid

import keras
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

import envs
import csb
import export_to_codingame_submission


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


def _restore_from_tf_checkpoint(filename, agent):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    with agent.model.graph.as_default():
        all_vars = tf.global_variables()
        for tensor_name in tensor_codes.values():
            matching_vars = [var for var in all_vars if var.op.name == tensor_name]
            assert len(matching_vars) == 1
            tensor_val = reader.get_tensor(tensor_name)
            matching_vars[0].load(tensor_val, agent.model.session)


def _restore_from_keras_checkpoint(filename, agent):
    tensor_values = {}
    with tf.Graph().as_default():
        model = keras.models.load_model(filename, compile=False)
        for tensor_key in tensor_codes.keys():
            layer_weights = model.get_layer(tensor_key.split('/')[0]).get_weights()
            tensor_weights = layer_weights[0 if tensor_key.endswith('/W') else 1]
            tensor_values[tensor_key] = tensor_weights

    with agent.model.graph.as_default():
        all_vars = tf.global_variables()
        for tensor_key, tensor_name in tensor_codes.items():
            matching_vars = [var for var in all_vars if var.op.name == tensor_name]
            assert len(matching_vars) == 1
            matching_vars[0].load(tensor_values[tensor_key], agent.model.session)


def _restore(filename, agent):
    if filename.endswith('.h5'):
        _restore_from_keras_checkpoint(filename, agent)
    else:
        _restore_from_tf_checkpoint(filename, agent)


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


def _avg(s):
    return sum(s) / len(s) if s else None


def do_train(gym_id, do_monitor, monitor_safe, monitor_video, agent_path, agent_kwargs, network_path,
             debug, timesteps, episodes, max_episode_timesteps, deterministic, load_path):

    monitor = '{}_{}_{}'.format(gym_id, _basename_no_ext(agent_path), _basename_no_ext(network_path))
    if deterministic:
        monitor += '_deterministic'

    environment = OpenAIGym(
        gym_id=gym_id,
        monitor='logs/{}/gym'.format(monitor) if do_monitor else None,
        monitor_safe=monitor_safe if do_monitor else None,
        monitor_video=monitor_video if do_monitor else None
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # log_levels[agent.log_level])

    if agent_path is not None:
        with open(agent_path, 'r') as fp:
            agent_json = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if network_path is not None:
        with open(network_path, 'r') as fp:
            network_json = json.load(fp=fp)
    else:
        network_json = None
        logger.info("No network configuration provided.")

    agent_json.update(agent_kwargs)
    agent = Agent.from_spec(
        spec=agent_json,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network_json,
            saver=dict(directory='logs/{}/checkpoints'.format(monitor), seconds=600),
            summarizer=dict(directory='logs/{}/summaries'.format(monitor),
                            labels=[],
                            seconds=120)
        )
    )

    logger.info("Starting agent for OpenAI Gym '{gym_id}'".format(gym_id=gym_id))
    logger.info("Config:")
    logger.info(agent)

    if load_path:
        _restore(load_path, agent)

    if debug:
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

    if debug:
        report_episodes = 1
    else:
        report_episodes = 100
        if agent_json.get('update_mode', {}).get('unit') == 'episodes':
            report_episodes = max(report_episodes, agent_json['update_mode'].get('frequency', 0))

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            logger.info("Latest episode rewards: {}".format(', '.join(map('{:.2f}'.format, r.episode_rewards[-5:]))))
            logger.info("All time best: {:0.2f}".format(max(r.episode_rewards)))
            logger.info("{} latest best: {:0.2f}".format(
                5 * report_episodes, max(r.episode_rewards[-5 * report_episodes:])))
            logger.info("Average of last {} rewards: {:0.2f}".format(
                20 * report_episodes, _avg(r.episode_rewards[-20 * report_episodes:])))
            logger.info("Average of last {} rewards: {:0.2f}".format(
                5 * report_episodes, _avg(r.episode_rewards[-5 * report_episodes:])))
            logger.info("Average of last {} rewards: {:0.2f}".format(
                report_episodes, _avg(r.episode_rewards[-report_episodes:])))
        if versus_opponent:
            versus_opponent.episode_finished(r.episode_rewards[-1])
        return True

    runner.run(
        timesteps=timesteps,
        episodes=episodes,
        max_episode_timesteps=max_episode_timesteps,
        deterministic=deterministic,
        episode_finished=episode_finished
    )
    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))
