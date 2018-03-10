import argparse
import os
import multiprocessing

import numpy as np
from tensorforce.contrib.openai_gym import OpenAIGym

import envs
from envs.csb.vincent_algo import VincentSalimInterface


def _generate_episode_data(episode_id, gym_id, monitor):
    if episode_id % 100 == 0:
        print('Computing game', episode_id)

    try:
        environment = OpenAIGym(
            gym_id=gym_id,
            monitor=monitor if episode_id == 0 else None,
            monitor_video=1 if episode_id == 0 else 0
        )
        state = environment.reset()
        world = environment.gym.unwrapped.world
        interface = VincentSalimInterface()
        interface.start(world)
        episode = []
        while True:
            interface.feed(world)
            actions = interface.get_moves(world, 0)
            action = [actions[0].g1, actions[0].g2, actions[0].g3, actions[1].g1, actions[1].g2, actions[1].g3]
            episode.append((state, action))
            state, terminal, step_reward = environment.execute(action)
            if terminal:
                break
        return episode
    except Exception:
        print('An exception occurred during game generation!')
        return []


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('--episodes', type=int, default=4000, help="Number of episodes")

    args = parser.parse_args()
    args.monitor = 'supervised_data_{}'.format(args.gym_id)

    with multiprocessing.Pool() as p:
        episodes = p.starmap(_generate_episode_data,
                             [(episode_id, args.gym_id, 'logs/{}/gym'.format(args.monitor))
                              for episode_id in range(args.episodes)],
                             chunksize=100)

    x = np.array([state for episode in episodes for state, _ in episode])
    y = np.array([action for episode in episodes for _, action in episode])

    save_dir = 'logs/{}'.format(args.monitor)
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(save_dir + '/data.npz', x=x, y=y)


if __name__ == '__main__':
    main()
