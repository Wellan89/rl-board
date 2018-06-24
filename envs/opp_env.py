import os
import gym

import numpy as np

DISABLE_RENDERING = bool(int(os.environ.get('DISABLE_RENDERING', 0)))


class OppEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'] if not DISABLE_RENDERING else []}
    reward_range = (-np.inf, np.inf)
    spec = None

    def __init__(self, policy_class):
        self.policy_class = policy_class
        self.timesteps = 0
        self.opponent_predict = None
        self.viewer = None

        self.hard_env_weight = 0.0
        self.opponent_factory = None

        self.action_space = None
        self.observation_space = None

    def create_new_instance(self):
        env = self.__class__()
        env.set_hard_env_weight(self.hard_env_weight)
        # assert env.raw_rewards_weight == self.raw_rewards_weight
        env.set_opponent_factory(self.opponent_factory)
        # assert env.opponent_factory is self.opponent_factory
        return env

    def reset(self):
        self.timesteps = 0
        self.opponent_predict = None
        return self._reset()

    def _reset(self):
        raise NotImplementedError('Abstract method')

    def _get_state(self, opponent_view=False):
        raise NotImplementedError('Abstract method')

    def _compute_opp_solution(self, dummy_opp_solution_func=None):
        if self.timesteps == 0 and self.opponent_factory:
            self.opponent_predict = self.opponent_factory()
        opponent_solution = None
        if self.opponent_predict:
            opponent_solution = self.opponent_predict(self._get_state(opponent_view=True))
        if opponent_solution is None:
            opponent_solution = dummy_opp_solution_func()
        return opponent_solution

    def step(self, action):
        raise NotImplementedError('Abstract method')

    def render(self, mode='human'):
        if DISABLE_RENDERING:
            return None

        self.viewer, rendered = self._render(viewer=self.viewer, mode=mode)
        return rendered

    def _render(self, viewer, mode):
        raise NotImplementedError('Abstract method')

    def set_hard_env_weight(self, weight):
        assert 0.0 <= weight <= 1.0
        self.hard_env_weight = weight

    def set_opponent_factory(self, opponent_factory):
        self.opponent_factory = opponent_factory


class OpponentPredictor:
    def __init__(self, env, weights, deterministic=True):
        self.policy_class = env.unwrapped.policy_class
        self.weights = weights
        self.deterministic = deterministic

    def __call__(self, state):
        # The None model is the default AI in the environment
        if self.weights:
            return self.policy_class.predict_from_weights(state=state, weights=self.weights, deterministic=self.deterministic)
        else:
            return None
