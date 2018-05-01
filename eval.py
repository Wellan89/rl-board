import argparse
import datetime
import math
import multiprocessing
import os

from scipy import stats
from tensorforce.contrib.openai_gym import OpenAIGym

import checkpoints_utils
import csb
import envs


def _generate_episode_data(episode_id, gym_id, model, versus_model, monitor):
    try:
        environment = OpenAIGym(
            gym_id=gym_id,
            monitor=monitor if episode_id == 0 else None,
            monitor_video=1 if episode_id == 0 else 0
        )
        if versus_model:
            environment.gym.unwrapped.opp_solution_predict = lambda _state, _: versus_model.predict(_state)
        state = environment.reset()
        reward = 0.0
        while True:
            action = model.predict(state)
            state, terminal, step_reward = environment.execute(action)
            reward += step_reward
            if terminal:
                break
        return reward
    except Exception:
        print('An exception occurred during game generation!')
        return 0.0


class _MappedGenerateEpisodeData:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, episode_id):
        return _generate_episode_data(episode_id, **self.kwargs)


def _compute_rewards(gym_id, model_path, episodes, monitor, processes,
                     deterministic, versus_model_path, versus_deterministic):
    with multiprocessing.Pool(processes=processes) as p:
        model = csb.Model(checkpoints_utils.read_weights(model_path), deterministic=deterministic)
        if versus_model_path:
            versus_model = csb.Model(checkpoints_utils.read_weights(versus_model_path), deterministic=versus_deterministic)
        else:
            versus_model = None
        return p.map(_MappedGenerateEpisodeData(gym_id=gym_id, model=model, versus_model=versus_model, monitor=monitor),
                     range(episodes))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-e', '--episodes', type=int, default=2000, help="Number of episodes")
    parser.add_argument('-l', '--load', help="Load agent from a previous checkpoint")
    parser.add_argument('-p', '--processes', type=int, default=None, help="Number of processes")
    parser.add_argument('-d', '--deterministic', type=int, default=1, help="Whether actions are chosen deterministically")
    parser.add_argument('-m', '--monitor', type=int, default=0, help="Whether to keep evaluating")
    parser.add_argument('-v', '--versus', help="Load versus agent from a previous checkpoint")
    parser.add_argument('-w', '--versus-deterministic', type=int, default=1,
                        help="Whether opponents actions are chosen deterministically")

    args = parser.parse_args()

    monitor = 'logs/eval_{}_{}'.format(args.gym_id, 'deterministic' if args.deterministic else 'random')
    eval_idx = 0
    while True:
        eval_idx += 1
        for model_path in args.load.split(','):
            print('{} - {} - Evaluation {} ({} run):'.format(model_path, datetime.datetime.now(), eval_idx,
                                                             'deterministic' if args.deterministic else 'random'))
            rewards = _compute_rewards(gym_id=args.gym_id, model_path=model_path, episodes=args.episodes,
                                       monitor=monitor, processes=args.processes, deterministic=bool(args.deterministic),
                                       versus_model_path=args.versus, versus_deterministic=bool(args.versus_deterministic))

            print('Rewards statistics:')
            desc = stats.describe(rewards)
            print(desc)
            print('95% confidence interval:', stats.norm.interval(0.95, loc=desc.mean, scale=math.sqrt(desc.variance / desc.nobs)))
            print('\n')

        if not args.monitor:
            break


if __name__ == '__main__':
    main()
