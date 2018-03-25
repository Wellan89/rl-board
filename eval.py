import argparse

from scipy import stats
from tensorforce.contrib.openai_gym import OpenAIGym
import tqdm

import checkpoints_utils
import csb
import envs


def _run_episode(environment, model):
    reward = 0.0
    state = environment.reset()
    while True:
        action = model.predict(state)
        state, terminal, step_reward = environment.execute(action)
        reward += step_reward
        if terminal:
            return reward


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-e', '--episodes', type=int, default=3000, help="Number of episodes")
    parser.add_argument('-l', '--load', default=None, help="Load agent from a previous checkpoint")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=1000, help="Save video every x steps (0 = disabled)")

    args = parser.parse_args()

    environment = OpenAIGym(
        gym_id=args.gym_id,
        monitor='logs/eval_{}'.format(args.gym_id),
        monitor_safe=args.monitor_safe,
        monitor_video=args.monitor_video
    )

    model = csb.Model(checkpoints_utils.read_weights(args.load))

    rewards = [_run_episode(environment, model) for _ in tqdm.trange(args.episodes)]
    print('Rewards statistics:')
    print(stats.describe(rewards))


if __name__ == '__main__':
    main()
