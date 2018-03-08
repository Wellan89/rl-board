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
    reward_range = (-np.inf, np.inf)
    spec = None

    difficulty_level = None
    variation = None

    def __init__(self):
        self.world = World()
        self.viewer = None

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32, shape=(6,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=(len(self._get_state()),))

    def _get_state(self):
        state = Observation(self.world, variation=self.variation).to_representation()
        # assert(all(self.observation_space.low <= v <= self.observation_space.high for v in state))
        return state

    def _transform_action(self, action):
        return action

    def step(self, action):
        # assert (len(action),) == self.action_space.shape
        # assert all(self.action_space.low <= v <= self.action_space.high for v in action)
        action = self._transform_action(action)
        assert len(action) == 6

        current_score = self.world.pods[0].score(variation=self.variation)
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
            now_score = self.world.pods[0].score(variation=self.variation)
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
        self.world = World()
        return self._get_state()

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

        def _pos_to_screen(_p):
            return _p.x * VIEWPORT_W / 16000, _p.y * VIEWPORT_H / 9000

        def _radius_to_screen(_r):
            return _r * VIEWPORT_W / 16000

        cp_radius = _radius_to_screen(self.world.circuit.cps[0].r + self.world.pods[0].r)
        for cp in self.world.circuit.cps:
            color_ratio = cp.id / (len(self.world.circuit.cps) - 1)
            color = (color_ratio * 0.8, color_ratio * 0.8, 0.2 + color_ratio * 0.8)
            self.viewer.draw_circle(color=color, radius=cp_radius).add_attr(
                rendering.Transform(translation=_pos_to_screen(cp))
            )

        pod_radius = _radius_to_screen(self.world.pods[0].r)
        for pod in self.world.pods:
            color = (float(pod.id >= 2), float(pod.id < 2), 0.0)
            if pod.shield > 0:
                self.viewer.draw_circle(color=(0.0, 0.0, 0.6),
                                        radius=pod_radius + _radius_to_screen(20 * pod.shield)).add_attr(
                    rendering.Transform(translation=_pos_to_screen(pod))
                )
            self.viewer.draw_circle(color=color, radius=pod_radius).add_attr(
                rendering.Transform(translation=_pos_to_screen(pod))
            )
            if pod.boost_available:
                self.viewer.draw_circle(color=(0.0, 0.0, 0.0),
                                        radius=_radius_to_screen(80)).add_attr(
                    rendering.Transform(translation=_pos_to_screen(pod))
                )

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


class CsbEnvD0V0(CsbEnv):
    difficulty_level = 0
    variation = 0


class CsbEnvD0Var1V0(CsbEnv):
    difficulty_level = 0
    variation = 1


class CsbEnvD0Var2V0(CsbEnv):
    difficulty_level = 0
    variation = 2


class CsbEnvD0Var3V0(CsbEnv):
    difficulty_level = 0
    variation = 3


class CsbEnvD0Var4V0(CsbEnv):
    difficulty_level = 0
    variation = 4

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32, shape=(10,))

    def _transform_action(self, action):
        assert len(action) == 10
        ret_action = []
        for i in range(0, 10, 5):
            pod_action = action[i:i+5]
            angle = 0.5 + 0.25 * (pod_action[0] - pod_action[1])
            speed = 0.2 + 0.6 * pod_action[2]
            if pod_action[3] > 0.5:
                boost_normal_or_shield = 0.0
            elif pod_action[4] > 0.5:
                boost_normal_or_shield = 1.0
            else:
                boost_normal_or_shield = 0.5
            ret_action += [angle, speed, boost_normal_or_shield]
        return ret_action


class CsbEnvD0Var5V0(CsbEnvD0Var4V0):
    difficulty_level = 0
    variation = 5


class CsbEnvD0Var6V0(CsbEnv):
    difficulty_level = 0
    variation = 6

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32, shape=(8,))

    def _transform_action(self, action):
        assert len(action) == 8
        ret_action = []
        for i in range(0, 8, 4):
            pod_action = action[i:i+4]
            angle = 0.5 + 0.25 * (pod_action[0] - pod_action[1])
            speed = 0.2 + 0.6 * pod_action[2]
            ret_action += [angle, speed, pod_action[3]]
        return ret_action


class CsbEnvD0Var7V0(CsbEnv):
    difficulty_level = 0
    variation = 7


class CsbEnvD1V0(CsbEnv):
    difficulty_level = 1
    variation = 0
