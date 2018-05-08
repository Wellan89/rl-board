import os
import gym
import numpy as np

from envs.csb.world import World

DISABLE_RENDERING = bool(int(os.environ.get('DISABLE_RENDERING', 0)))


class CsbEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'] if not DISABLE_RENDERING else []}
    reward_range = (-np.inf, np.inf)
    spec = None

    def __init__(self):
        self.world = World()
        self.is_new_episode = True
        self.viewer = None

        self.raw_rewards_weight = 0.0
        self.opp_solution_predict = None

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32, shape=(6,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=(len(self._get_state()),))

    def reset(self):
        self.world = World()
        self.is_new_episode = True
        return self._get_state()

    def _get_state(self, opponent_view=False):
        return self.world.compute_state(opponent_view=opponent_view)

    def step(self, action):
        # assert (len(action),) == self.action_space.shape
        # assert all(self.action_space.low <= v <= self.action_space.high for v in action)
        action = np.clip(action, 0.0, 1.0)

        last_score = self.world.compute_agent_score()

        opp_solution = None
        if self.opp_solution_predict is not None:
            opp_solution = self.opp_solution_predict(self._get_state(opponent_view=True), self.is_new_episode)
        self.is_new_episode = False

        # Dummy solution : straight line toward the next checkpoint
        if opp_solution is None:
            opp_solution = self.world.dummy_opp_solution()

        safe_state = self._get_state()
        try:
            self.world.play(action, opp_solution)
        except Exception as e:
            print('An error occurred during game generation!', e)
            return safe_state, 0.0, True, {}

        players_won = [self.world.player_won(i) for i in range(2)]
        if players_won[0] and players_won[1]:
            episode_over = True
            easy_reward = 0.0
            raw_reward = 0.0
        elif players_won[1]:
            episode_over = True
            easy_reward = 0.0
            raw_reward = -10.0
        elif players_won[0]:
            episode_over = True
            easy_reward = 0.0
            raw_reward = 10.0
        else:
            episode_over = False
            easy_reward = self.world.compute_agent_score() - last_score
            raw_reward = 0.0

        reward = (1.0 - self.raw_rewards_weight) * easy_reward + self.raw_rewards_weight * raw_reward
        # assert self.reward_range[0] <= reward <= self.reward_range[1]
        return self._get_state(), reward, episode_over, {}

    def render(self, mode='human'):
        if DISABLE_RENDERING:
            return None

        self.viewer, rendered = self.world.render(viewer=self.viewer, mode=mode)
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
