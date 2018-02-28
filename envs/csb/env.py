import gym
import numpy as np
from gym.envs.classic_control import rendering

from envs.csb.world import World
from envs.csb.solution import Solution
from envs.csb.move import Move
from envs.csb.observation import Observation

VIEWPORT_W = 600
VIEWPORT_H = 400


class CsbEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (0, 10.0)
    spec = None

    action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-100.0, high=100.0,
                                       shape=(len(Observation(World()).to_representation()),), dtype=np.float32)

    def __init__(self):
        self.world = World()
        self.viewer = None

    def _get_state(self):
        return Observation(self.world).to_representation()

    def step(self, action):
        assert len(action) == 2 and all(0 <= v <= 1 for v in action)

        current_score = self.world.pods[0].score()

        self.world.play(
            Solution(
                move1=Move(
                    g1=action[0],
                    g2=action[1]/2,
                    g3=0.5,
                ),
                move2=Move(
                    g1=0.5,
                    g2=0,
                    g3=0.5,
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

        now_score = self.world.pods[0].score()
        reward = (now_score - current_score)
        episode_over = False

        return self._get_state(), reward, episode_over, None

    def reset(self):
        self.world.reset()
        return self._get_state()

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

        def _pos_to_screen(_p):
            return _p.x * VIEWPORT_W / 16000, _p.y * VIEWPORT_H / 9000

        for i, pod in enumerate(self.world.pods):
            self.viewer.draw_circle(color=(int(i >= 2), int(i < 2), 0)).add_attr(
                rendering.Transform(translation=_pos_to_screen(pod))
            )

        for i, cp in enumerate(self.world.circuit.cps):
            self.viewer.draw_circle(color=(0, 0, 1), radius=10+3*(i+1)).add_attr(
                rendering.Transform(translation=_pos_to_screen(cp))
            )

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
