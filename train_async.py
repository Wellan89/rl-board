"""
OpenAI gym execution

To run this script with 3 workers on Pong-ram-v0:
$ python train_async.py Pong-ram-v0 -a agents/trpo-v0.json -n networks/mlp-v1.json -W 3

You can check what the workers are doing:
$ tmux a -t OpenAI  # `ctrl+b d` to exit tmux

To kill the session:
$ python train_async.py Pong-ram-v0 -W 3 -K
"""

import argparse
import inspect
import json
import logging
import os
import sys
import time

import tensorflow as tf
from six.moves import shlex_quote

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

import envs


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
    parser.add_argument('-d', '--deterministic', action='store_true', help="Choose actions deterministically")
    parser.add_argument('-M', '--mode', choices=('tmux', 'child'), default='tmux', help="Starter mode")
    parser.add_argument('-W', '--num-workers', type=int, default=1, help="Number of worker agents")
    parser.add_argument('-C', '--child', action='store_true', help="Child process")
    parser.add_argument('-P', '--parameter-server', action='store_true', help="Parameter server")
    parser.add_argument('-I', '--task-index', type=int, default=0, help="Task index")
    parser.add_argument('-K', '--kill', action='store_true', help="Kill runners")
    parser.add_argument('-L', '--logdir', default='logs_async', help="Log directory")
    parser.add_argument('-D', '--debug', action='store_true', help="Show debug outputs")
    parser.add_argument('--monitor', help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=1000, help="Save video every x steps (0 = disabled)")

    args = parser.parse_args()

    if not args.monitor:
        args.monitor = '{}_{}_{}'.format(args.gym_id, _basename_no_ext(args.agent), _basename_no_ext(args.network))

    session_name = 'OpenAI-' + args.gym_id
    shell = '/bin/bash'

    kill_cmds = [
        "kill $( lsof -i:12222-{} -t ) > /dev/null 2>&1".format(12222 + args.num_workers),
        "tmux kill-session -t {}".format(session_name),
    ]
    if args.kill:
        os.system("\n".join(kill_cmds))
        return 0

    if not args.child:
        # start up child processes
        target_script = os.path.abspath(inspect.stack()[0][1])

        def wrap_cmd(session, name, cmd):
            if isinstance(cmd, list):
                cmd = ' '.join(shlex_quote(str(arg)) for arg in cmd)
            if args.mode == 'tmux':
                return 'tmux send-keys -t {}:{} {} Enter'.format(session, name, shlex_quote(cmd))
            elif args.mode == 'child':
                return '{} > {}/{}.{}.out 2>&1 & echo kill $! >> {}/kill.sh'.format(
                    cmd, args.logdir, session, name, args.logdir
                )

        def build_cmd(ps, index):
            cmd_args = [
                'CUDA_VISIBLE_DEVICES=',
                sys.executable, target_script,
                args.gym_id,
                '--agent', os.path.join(os.getcwd(), args.agent),
                '--network', os.path.join(os.getcwd(), args.network),
                '--num-workers', args.num_workers,
                '--child',
                '--task-index', index
            ]
            if args.episodes is not None:
                cmd_args.append('--episodes')
                cmd_args.append(args.episodes)
            if args.timesteps is not None:
                cmd_args.append('--timesteps')
                cmd_args.append(args.timesteps)
            if args.max_episode_timesteps is not None:
                cmd_args.append('--max-episode-timesteps')
                cmd_args.append(args.max_episode_timesteps)
            if args.deterministic:
                cmd_args.append('--deterministic')
            if ps:
                cmd_args.append('--parameter-server')
            if args.debug:
                cmd_args.append('--debug')
            return cmd_args

        if args.mode == 'tmux':
            cmds = kill_cmds + ['tmux new-session -d -s {} -n ps'.format(session_name)]
        else:
            assert args.mode == 'child'
            cmds = ['mkdir -p {}'.format(args.logdir),
                    'rm -f {}/kill.sh'.format(args.logdir),
                    'echo "#/bin/bash" > {}/kill.sh'.format(args.logdir),
                    'chmod +x {}/kill.sh'.format(args.logdir)]

        cmds.append(wrap_cmd(session_name, 'ps', build_cmd(ps=True, index=0)))

        for i in range(args.num_workers):
            name = 'worker{}'.format(i)
            if args.mode == 'tmux':
                cmds.append('tmux new-window -t {} -n {} -d {}'.format(session_name, name, shell))
            cmds.append(wrap_cmd(session_name, name, build_cmd(ps=False, index=i)))

        # add one PS call
        # cmds.append('tmux new-window -t {} -n ps -d {}'.format(session_name, shell))

        print("\n".join(cmds))

        os.system("\n".join(cmds))

        return 0

    ps_hosts = ['127.0.0.1:{}'.format(12222)]
    worker_hosts = []
    port = 12223
    for _ in range(args.num_workers):
        worker_hosts.append('127.0.0.1:{}'.format(port))
        port += 1
    cluster = {'ps': ps_hosts, 'worker': worker_hosts}
    cluster_spec = tf.train.ClusterSpec(cluster)

    do_monitor = True  # (args.task_index == 0)
    environment = OpenAIGym(
        gym_id=args.gym_id,
        monitor='logs/{}/gym'.format(args.monitor) if do_monitor and args.monitor else None,
        monitor_safe=args.monitor_safe if do_monitor else None,
        monitor_video=args.monitor_video // args.num_workers if do_monitor and args.monitor_video else None
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # log_levels[agent.log_level])

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

    if args.parameter_server:
        agent['device'] = '/job:ps/task:{}'.format(args.task_index)  # '/cpu:0'
    else:
        agent['device'] = '/job:worker/task:{}'.format(args.task_index)  # '/cpu:0'

    agent['distributed'] = dict(
        cluster_spec=cluster_spec,
        task_index=args.task_index,
        parameter_server=args.parameter_server,
        protocol='grpc'
    )

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network,
            saver=dict(directory='logs/{}/checkpoints'.format(args.monitor), seconds=600),
            summarizer=dict(directory='logs/{}/summaries'.format(args.monitor),
                            labels=[],
                            seconds=120)
        )
    )

    logger.info("Starting distributed agent for OpenAI Gym '{gym_id}'".format(gym_id=args.gym_id))
    logger.info("Config:")
    logger.info(agent)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    if args.debug:
        report_episodes = 1
    else:
        report_episodes = 100

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {:d} after overall {:d} timesteps. Steps Per Second {:.2f}".format(
                r.agent.episode,
                r.agent.timestep,
                steps_per_second)
            )
            logger.info("Latest episode rewards: {}".format(', '.join(map('{:.2f}'.format, r.episode_rewards[-5:]))))
            logger.info("All time best: {:0.2f}".format(max(r.episode_rewards)))
            logger.info("Average of last 500 rewards: {:.2f}".format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {:.2f}".format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
        return True

    runner.run(
        timesteps=args.timesteps,
        episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished
    )
    runner.close()


if __name__ == '__main__':
    main()
