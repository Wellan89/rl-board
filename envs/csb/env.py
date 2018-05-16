import os
import gym

from csb import csb

import numpy as np
from envs.csb import renderer

DISABLE_RENDERING = bool(int(os.environ.get('DISABLE_RENDERING', 0)))


class CsbEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'] if not DISABLE_RENDERING else []}
    reward_range = (-np.inf, np.inf)
    spec = None

    def __init__(self):
        self.world = None
        self.timesteps = 0
        self.reset()
        self.viewer = None

        self.raw_rewards_weight = 0.0
        self.opp_solution_predict = None

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32, shape=(6,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=(len(self._get_state()),))

    @classmethod
    def create_new_instance(cls):
        return cls()

    def reset(self):
        self.world = csb.World()
        self.timesteps = 0
        return self._get_state()

    def _get_state(self, opponent_view=False):
        return np.array(self.world.compute_state(opponent_view))

    def compute_dense_score(self):
        return 0.5 * sum(pod.score() for pod in self.world.pods[:2])

    def compute_distance_score(self):
        return max(pod.score() for pod in self.world.pods[:2]) - max(pod.score() for pod in self.world.pods[2:])

    def blocking_score(self):
        return -50.0 * max(pod.nb_checked() for pod in self.world.pods[2:]) / self.timesteps

    def step(self, action):
        # assert (len(action),) == self.action_space.shape
        # assert all(self.action_space.low <= v <= self.action_space.high for v in action)
        action = np.clip(action, 0.0, 1.0)

        last_score = self.compute_dense_score()
        last_distance_score = self.compute_distance_score()

        opp_solution = None
        if self.opp_solution_predict is not None:
            opp_solution = self.opp_solution_predict(self._get_state(opponent_view=True), (self.timesteps == 0))

        # Dummy solution : straight line toward the next checkpoint
        if opp_solution is None:
            opp_solution = self.world.dummy_opp_solution()

        safe_state = self._get_state()
        try:
            self.world.play(action, opp_solution)

            state = self._get_state()
            nans = [np.any(np.isnan(action)), np.any(np.isnan(opp_solution)), np.any(np.isnan(state))]
            if any(nans):
                print('NaN value found:', nans)
                print('action:', action)
                print('opp_solution:', opp_solution)
                print('state:', state)
                raise RuntimeError('NaN value found')

        except Exception as e:
            print('An error occurred during game generation!', e)
            return safe_state, 0.0, True, {}
        self.timesteps += 1

        easy_reward = self.compute_dense_score() - last_score
        distance_score = self.compute_distance_score() - last_distance_score
        players_won = [self.world.player_won(i) for i in range(2)]
        if players_won[0] and players_won[1]:
            episode_over = True
            raw_reward = 0.0
        elif players_won[1]:
            episode_over = True
            raw_reward = -10.0
        elif players_won[0]:
            episode_over = True
            raw_reward = 10.0
        else:
            episode_over = False
            raw_reward = 0.0

        raw_reward += distance_score
        if episode_over:
            raw_reward += self.blocking_score()

        reward = (1.0 - self.raw_rewards_weight) * easy_reward + self.raw_rewards_weight * raw_reward
        # assert self.reward_range[0] <= reward <= self.reward_range[1]
        return state, reward, episode_over, {}

    def render(self, mode='human'):
        if DISABLE_RENDERING:
            return None

        self.viewer, rendered = renderer.render(world=self.world, viewer=self.viewer, mode=mode)
        return rendered

    def set_hard_env_weight(self, weight):
        assert 0.0 <= weight <= 1.0
        self.raw_rewards_weight = weight

    def enable_opponent(self, opp_solution_predict):
        self.opp_solution_predict = opp_solution_predict


class CsbEnvD0(CsbEnv):
    pass


class CsbEnvD1(CsbEnv):
    def __init__(self):
        super().__init__()
        self.set_hard_env_weight(weight=1.0)
