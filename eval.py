import argparse
import math
import multiprocessing

from scipy import stats
from tensorforce.contrib.openai_gym import OpenAIGym

import checkpoints_utils
import csb
import envs


def _generate_episode_data(episode_id, gym_id, model, monitor):
    try:
        environment = OpenAIGym(
            gym_id=gym_id,
            monitor=monitor if episode_id == 0 else None,
            monitor_video=1 if episode_id == 0 else 0
        )
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
    def __init__(self, gym_id, model, monitor):
        self.gym_id = gym_id
        self.model = model
        self.monitor = monitor

    def __call__(self, episode_id):
        return _generate_episode_data(episode_id, self.gym_id, self.model, self.monitor)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-e', '--episodes', type=int, default=2000, help="Number of episodes")
    parser.add_argument('-l', '--load', default=None, help="Load agent from a previous checkpoint")

    args = parser.parse_args()

    with multiprocessing.Pool() as p:
        rewards = p.map(_MappedGenerateEpisodeData(gym_id=args.gym_id,
                                                   model=csb.Model(checkpoints_utils.read_weights(args.load)),
                                                   monitor='logs/eval_{}'.format(args.gym_id)),
                        range(args.episodes))

    print('Rewards statistics:')
    desc = stats.describe(rewards)
    print(desc)
    print('95% confidence interval:', stats.norm.interval(0.95, loc=desc.mean, scale=math.sqrt(desc.variance / desc.nobs)))


if __name__ == '__main__':
    main()
