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

    use_scored_reward = None
    use_timed_features_mask = False
    use_cp_dist_score = False

    def __init__(self):
        self.world = World()
        self.viewer = None

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32, shape=(6,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=(len(self._get_state()),))

    def _get_state(self):
        state = Observation(self.world, use_timed_features_mask=self.use_timed_features_mask).to_representation()
        # assert(all(self.observation_space.low <= v <= self.observation_space.high for v in state))
        return state

    def _transform_action(self, action):
        return action

    def step(self, action):
        # assert (len(action),) == self.action_space.shape
        # assert all(self.action_space.low <= v <= self.action_space.high for v in action)
        action = self._transform_action(action)
        assert len(action) == 6

        current_score = self.world.pods[0].score(use_cp_dist_score=self.use_cp_dist_score)
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

        if self.use_scored_reward:
            now_score = self.world.pods[0].score(use_cp_dist_score=self.use_cp_dist_score)
            reward = now_score - current_score
            episode_over = (self.world.turn >= 400)
        else:
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
    use_scored_reward = True
    use_cp_dist_score = True
    use_timed_features_mask = True


class CsbEnvD1V0(CsbEnv):
    use_scored_reward = True
    use_cp_dist_score = False
    use_timed_features_mask = False


class CsbEnvD2V0(CsbEnv):
    use_scored_reward = False
    use_cp_dist_score = False
    use_timed_features_mask = False
