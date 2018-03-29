import argparse
import datetime
import math

from scipy import stats

import eval


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-e', '--episodes', type=int, default=2000, help="Number of episodes")
    parser.add_argument('-l', '--load', default=None, help="Load agent from a previous checkpoint")
    parser.add_argument('-p', '--processes', type=int, default=None, help="Number of processes")

    args = parser.parse_args()

    eval_idx = 0
    while True:
        eval_idx += 1
        print('{} - Evaluation {}:'.format(datetime.datetime.now(), eval_idx))
        rewards = eval.compute_rewards(gym_id=args.gym_id, model_path=args.load,
                                       episodes=args.episodes, processes=args.processes)

        print('Rewards statistics:')
        desc = stats.describe(rewards)
        print(desc)
        print('95% confidence interval:', stats.norm.interval(0.95, loc=desc.mean, scale=math.sqrt(desc.variance / desc.nobs)))
        print('\n' * 2)


if __name__ == '__main__':
    main()
