import gym
import numpy as np

from envs.csb.world import World
from envs.csb.solution import Solution
from envs.csb.move import Move
from envs.csb.observation import Observation


class CsbEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-100.0, 100.0)
    spec = None

    action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
    observation_space = gym.spaces.Box(shape=(len(Observation(World()).to_representation()),), dtype=np.float32)

    def __init__(self):
        self.world = World()

    def step(self, action):
        self._check_action_format(action)

        current_passed_cp = self.world.best_pod(0).nbChecked()

        self.world.play(
            Solution(
                move1=Move(
                    g1=action[0][0],
                    g2=action[0][1],
                    g3=action[0][2],
                ),
                move2=Move(
                    g1=action[1][0],
                    g2=action[1][1],
                    g3=action[1][2],
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
            reward = -100
        elif self.world.player_won(0):
            episode_over = True
            reward = 100
        else:
            now_passed_cp = self.world.best_pod(0).nbChecked()
            assert now_passed_cp >= current_passed_cp
            reward = now_passed_cp - current_passed_cp
            episode_over = False

        return Observation(self.world), reward, episode_over, self.get_debug()

    def _check_action_format(self, action):
        assert len(action) == 2
        for pod_action in action:
            assert len(pod_action) == 3
            for gene in pod_action:
                assert 0 <= gene <= 1

    def reset(self):
        self.world.reset()

    def render(self, mode='human', close=False):
        pass

    def get_debug(self):
        return {}
