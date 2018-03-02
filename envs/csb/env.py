import gym
import numpy as np
from gym.envs.classic_control import rendering

from envs.csb.world import World
from envs.csb.solution import Solution
from envs.csb.move import Move
from envs.csb.observation import Observation

VIEWPORT_W = 710
VIEWPORT_H = 400


class CsbEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-10.0, 10.0)
    spec = None

    action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-100.0, high=100.0,
                                       shape=(len(Observation(World()).to_representation()),), dtype=np.float32)

    difficulty_level = None

    def __init__(self):
        self.world = World()
        self.viewer = None

    def _get_state(self):
        state = Observation(self.world).to_representation()
        # assert(all(-100.0 <= v <= 100.0 for v in state))
        return state

    def step(self, action):
        assert len(action) == 6 and all(0.0 <= v <= 1.0 for v in action)

        current_score = self.world.pods[0].score()
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

        if self.difficulty_level == 0:
            now_score = self.world.best_pod(0).score()
            #reward = max(now_score - current_score, 0.0)# to test
            reward = now_score - current_score
            episode_over = (self.world.turn >= 400)
        elif self.difficulty_level == 1:
            if self.world.player_won(1):
                episode_over = True
                reward = -10.0
            elif self.world.player_won(0):
                episode_over = True
                reward = 10.0
            else:
                now_passed_cp = max(map(lambda pod: pod.nbChecked(), self.world.pods[:2]))
                assert now_passed_cp >= current_passed_cp
                reward = (now_passed_cp - current_passed_cp) * 0.1
                episode_over = False
        else:
            raise ValueError('Unknown difficulty level: {}'.format(self.difficulty_level))

        # assert self.reward_range[0] <= reward <= self.reward_range[1]
        return self._get_state(), reward, episode_over, None

    def reset(self):
        self.world.reset()
        return self._get_state()

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

        def _pos_to_screen(_p):
            return _p.x * VIEWPORT_W / 16000, _p.y * VIEWPORT_H / 9000

        def _radius_to_screen(_r):
            return _r * VIEWPORT_W / 16000

        for cp in self.world.circuit.cps:
            color_ratio = cp.id / (len(self.world.circuit.cps) - 1)
            color = (color_ratio * 0.8, color_ratio * 0.8, 0.2 + color_ratio * 0.8)
            self.viewer.draw_circle(color=color, radius=_radius_to_screen(cp.r)).add_attr(
                rendering.Transform(translation=_pos_to_screen(cp))
            )

        for i, pod in enumerate(self.world.pods):
            self.viewer.draw_circle(color=(int(i >= 2), int(i < 2), 0), radius=_radius_to_screen(pod.r)).add_attr(
                rendering.Transform(translation=_pos_to_screen(pod))
            )

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


class CsbEnvV0D0(CsbEnv):
    difficulty_level = 0


class CsbEnvV0D1(CsbEnv):
    difficulty_level = 1
