import random

import gym
import numpy as np

import stc_agent
from cpp import stc_pybind
from envs import opp_env
from envs.stc import renderer

stc_pybind.srand(random.SystemRandom().getrandbits(32))


class StcEnv(opp_env.OppEnv):
    opponent_simulations = -1

    def __init__(self):
        super().__init__(model_class=stc_agent.Model)
        self.world = None

        self.reset()

        self.action_space = gym.spaces.Discrete(n=22)
        #self.observation_space = gym.spaces.MultiDiscrete(nvec=[7] * (8 * 2) + [7] * (12 * 6 * 2))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=[8 * 2 + 12 * 6 * 2], dtype=np.float32)

    def _reset(self):
        self.world = stc_pybind.World()
        return self._get_state()

    def _get_state(self, opponent_view=False):
        return np.array(self.world.compute_state(opponent_view))

    def step(self, action):
        opp_solution = self._compute_opp_solution(
            dummy_opp_solution_func=lambda: self.world.dummy_opp_solution(self.opponent_simulations)
        )

        easy_reward = self.world.play(action, opp_solution)
        self.timesteps += 1

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

        reward = (1.0 - self.hard_env_weight) * easy_reward + self.hard_env_weight * raw_reward
        return self._get_state(), reward, episode_over, {}

    def _render(self, viewer, mode):
        return renderer.render(world=self.world, viewer=viewer, mode=mode)


class StcEnvD0(StcEnv):
    pass


class StcEnvD1(StcEnv):
    def __init__(self):
        super().__init__()
        self.set_hard_env_weight(weight=1.0)


class StcEnvD2(StcEnv):
    opponent_simulations = 1000


class StcEnvD3(StcEnvD2):
    def __init__(self):
        super().__init__()
        self.set_hard_env_weight(weight=1.0)
