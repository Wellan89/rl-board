import gym
import numpy as np
from gym.envs.classic_control import rendering

from envs.csb import observation
from envs.csb.world import World
from envs.csb.solution import Solution
from envs.csb.move import Move

VIEWPORT_W = 710
VIEWPORT_H = 400


class CsbEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-np.inf, np.inf)
    spec = None

    opp_solution_predict = None

    use_cp_dist_score = False
    use_raw_rewards = False
    use_negative_rewards = False
    use_timed_features_mask = False
    dummy_opponent_speed = 0.0
    versus_opponent_update_reward_threshold = 0.0

    def __init__(self):
        self.world = World()
        self.viewer = None

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32, shape=(6,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=(len(self._get_state()),))

    def _get_state(self, opponent_view=False):
        if opponent_view:
            pods = self.world.pods[2:] + self.world.pods[:2]
        else:
            pods = self.world.pods

        state = observation.observation(self.world, pods, use_timed_features_mask=self.use_timed_features_mask)
        # assert(all(self.observation_space.low <= v <= self.observation_space.high for v in state))
        return state

    def _action_to_solution(self, action):
        assert len(action) == 6
        return Solution(
            move1=Move(g1=action[0], g2=action[1], g3=action[2]),
            move2=Move(g1=action[3], g2=action[4], g3=action[5]),
        )

    def step(self, action):
        # assert (len(action),) == self.action_space.shape
        # assert all(self.action_space.low <= v <= self.action_space.high for v in action)

        best_pod = max(self.world.pods[:2], key=lambda pod: pod.score(use_cp_dist_score=self.use_cp_dist_score))
        current_score = best_pod.score(use_cp_dist_score=self.use_cp_dist_score)
        opp_current_score = max(pod.score(use_cp_dist_score=self.use_cp_dist_score) for pod in self.world.pods[2:])

        agent_solution = self._action_to_solution(action)

        # Dummy solution : straight line toward the next checkpoint
        if self.versus_opponent_update_reward_threshold == 0.0:
            opp_solution = Solution(
                move1=self.world.pods[2].to_dummy_move(speed=self.dummy_opponent_speed),
                move2=self.world.pods[3].to_dummy_move(speed=self.dummy_opponent_speed),
            )
        else:
            opp_action = self.opp_solution_predict(self._get_state(opponent_view=True))
            opp_solution = self._action_to_solution(opp_action)

        self.world.play(agent_solution, opp_solution)

        enable_timeout = (self.dummy_opponent_speed > 0.0 or self.versus_opponent_update_reward_threshold != 0.0)
        if self.world.player_won(1, enable_timeout=enable_timeout):
            episode_over = True
            reward = 0.0 if not self.use_negative_rewards else -10.0
        elif self.world.player_won(0, enable_timeout=enable_timeout):
            episode_over = True
            reward = 20.0 if not self.use_negative_rewards else 10.0
        else:
            if not self.use_raw_rewards:
                now_score = best_pod.score(use_cp_dist_score=self.use_cp_dist_score)
                opp_now_score = max(pod.score(use_cp_dist_score=self.use_cp_dist_score) for pod in self.world.pods[2:])
                reward = now_score - current_score - 0.1 * (opp_now_score - opp_current_score)
            else:
                reward = 0.0
            episode_over = False if enable_timeout else (self.world.turn > 500)

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
            self.viewer.draw_line(
                color=color,
                start=_pos_to_screen(pod),
                end=_pos_to_screen(pod.next_checkpoint(self.world)),
            )

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


class CsbEnvD0V0(CsbEnv):
    use_cp_dist_score = True
    use_timed_features_mask = True
    dummy_opponent_speed = 0.0


class CsbEnvD1V0(CsbEnv):
    use_cp_dist_score = True
    dummy_opponent_speed = 0.1


class CsbEnvD2V0(CsbEnv):
    use_cp_dist_score = True
    dummy_opponent_speed = 0.4


class CsbEnvVersusV0(CsbEnv):
    use_cp_dist_score = False
    use_raw_rewards = True
    use_negative_rewards = True
    dummy_opponent_speed = 0.0
    versus_opponent_update_reward_threshold = 2.0  # 60% of games won
