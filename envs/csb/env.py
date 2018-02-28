import gym
import numpy as np

from envs.csb.world import World
from envs.csb.solution import Solution
from envs.csb.move import Move
from envs.csb.observation import Observation


class CsbEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-10.0, 10.0)
    spec = None

    action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-100.0, high=100.0,
                                       shape=(len(Observation(World()).to_representation()),), dtype=np.float32)

    def __init__(self):
        self.world = World()

    def _get_state(self):
        return Observation(self.world).to_representation()

    def step(self, action):
        assert len(action) == 6 and all(0 <= v <= 1 for v in action)

        current_passed_cp = self.world.best_pod(0).nbChecked()

        self.world.play(
            Solution(
                move1=Move(
                    g1=action[0],
                    g2=action[1],
                    g3=action[2],
                ),
                move2=Move(
                    g1=action[3],
                    g2=action[4],
                    g3=action[5],
                )
            ),
            Solution(  # Placeholder : enemy doesn't move
                move1=Move(
                    g1=0.5,
                    g2=0,
                    g3=0.5,
                ),
                move2=Move(
                    g1=0.5,
                    g2=0,
                    g3=0.5,
                ),
            )
        )

        if self.world.player_won(1):
            episode_over = True
            reward = -10.0
        elif self.world.player_won(0):
            episode_over = True
            reward = 10.0
        else:
            now_passed_cp = self.world.best_pod(0).nbChecked()
            assert now_passed_cp >= current_passed_cp
            reward = (now_passed_cp - current_passed_cp) * 0.1
            episode_over = False

        return self._get_state(), reward, episode_over, None

    def reset(self):
        self.world.reset()
        return self._get_state()

    def render(self, mode='human', close=False):
        pass
