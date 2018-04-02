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
import os
import sys

import tensorflow as tf
from six.moves import shlex_quote

import train_utils


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
    parser.add_argument('-P', '--num-parameter-servers', type=int, default=1, help="Number of parameter servers")
    parser.add_argument('-C', '--child', action='store_true', help="Child process")
    parser.add_argument('--parameter-server', action='store_true', help="Parameter server")
    parser.add_argument('-I', '--task-index', type=int, default=0, help="Task index")
    parser.add_argument('-K', '--kill', action='store_true', help="Kill runners")
    parser.add_argument('-L', '--logdir', default='logs_async', help="Log directory")
    parser.add_argument('-D', '--debug', action='store_true', help="Show debug outputs")
    parser.add_argument('--monitor', help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=1000, help="Save video every x steps (0 = disabled)")
    parser.add_argument('-l', '--load', default=None, help="Load agent from a previous checkpoint")

    args = parser.parse_args()

    session_name = 'OpenAI-' + args.gym_id
    shell = '/bin/bash'

    base_port = 12222
    kill_cmds = [
        "kill $( lsof -i:{}-{} -t ) > /dev/null 2>&1".format(base_port,
                                                             base_port + args.num_parameter_servers + args.num_workers),
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
                '--num-parameter-servers', args.num_parameter_servers,
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
            if args.load:
                cmd_args.append('--load')
                cmd_args.append(args.load)
            return cmd_args

        if args.mode == 'tmux':
            cmds = kill_cmds + ['tmux new-session -d -s {}'.format(session_name)]
        else:
            assert args.mode == 'child'
            cmds = ['mkdir -p {}'.format(args.logdir),
                    'rm -f {}/kill.sh'.format(args.logdir),
                    'echo "#/bin/bash" > {}/kill.sh'.format(args.logdir),
                    'chmod +x {}/kill.sh'.format(args.logdir)]

        for is_ps, num in [(True, args.num_parameter_servers), (False, args.num_workers)]:
            for i in range(num):
                name = '{}{}'.format('ps' if is_ps else 'worker', i)
                if args.mode == 'tmux':
                    cmds.append('tmux new-window -t {} -n {} -d {}'.format(session_name, name, shell))
                cmds.append(wrap_cmd(session_name, name, build_cmd(ps=is_ps, index=i)))

        print("\n".join(cmds))
        os.system("\n".join(cmds))

        return 0

    port = base_port
    cluster = {'ps': [], 'worker': []}
    for is_ps, num in [(True, args.num_parameter_servers), (False, args.num_workers)]:
        for _ in range(num):
            cluster['ps' if is_ps else 'worker'].append('127.0.0.1:{}'.format(port))
            port += 1

    agent_kwargs = dict(
        device='/job:{}/task:{}'.format('ps' if args.parameter_server else 'worker', args.task_index),
        distributed=dict(
            cluster_spec=tf.train.ClusterSpec(cluster),
            task_index=args.task_index,
            parameter_server=args.parameter_server,
            protocol='grpc'
        )
    )

    train_utils.do_train(
        gym_id=args.gym_id,
        do_monitor=(args.task_index == 0),
        monitor_safe=args.monitor_safe,
        monitor_video=args.monitor_video,
        agent_path=args.agent,
        agent_kwargs=agent_kwargs,
        network_path=args.network,
        debug=args.debug,
        timesteps=args.timesteps,
        episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        load_path=args.load,
        task_index=args.task_index,
    )


if __name__ == '__main__':
    main()
