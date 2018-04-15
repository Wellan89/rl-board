import argparse

import train_utils


# python train.py csb-d0-v0 -a agents/trpo-v1.json -n networks/mlp-v1.json


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

    train_utils.do_train(
        gym_id=args.gym_id,
        do_monitor=True,
        monitor_safe=args.monitor_safe,
        monitor_video=args.monitor_video,
        agent_path=args.agent,
        agent_kwargs={},
        network_path=args.network,
        debug=args.debug,
        timesteps=args.timesteps,
        episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        load_path=args.load,
        task_index=None,
    )


if __name__ == '__main__':
    main()
