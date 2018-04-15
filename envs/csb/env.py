import os
import gym
import numpy as np

from envs.csb import observation
from envs.csb.world import World
from envs.csb.solution import Solution
from envs.csb.move import Move

DISABLE_RENDERING = bool(int(os.environ.get('DISABLE_RENDERING', 1)))


class CsbEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'] if not DISABLE_RENDERING else []}
    reward_range = (-np.inf, np.inf)
    spec = None

    def __init__(self):
        self.world = World()
        self.viewer = None

        self.use_cp_dist_score = True
        self.use_raw_rewards = False
        self.opp_solution_predict = None

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32, shape=(6,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=(len(self._get_state()),))

    def compute_custom_state(self, world, opponent_view=False):
        if opponent_view:
            pods = world.pods[2:] + world.pods[:2]
        else:
            pods = world.pods

        state = observation.observation(world, pods)
        # assert(all(self.observation_space.low <= v <= self.observation_space.high for v in state))
        return np.array(state)

    def _get_state(self, opponent_view=False):
        return self.compute_custom_state(self.world, opponent_view=opponent_view)

    def _action_to_solution(self, action):
        assert len(action) == 6
        return Solution(
            move1=Move(g1=action[0], g2=action[1], g3=action[2]),
            move2=Move(g1=action[3], g2=action[4], g3=action[5]),
        )

    def step(self, action):
        # assert (len(action),) == self.action_space.shape
        # assert all(self.action_space.low <= v <= self.action_space.high for v in action)

        action = np.clip(action, 0.0, 1.0)

        if not self.use_raw_rewards:
            best_pod = max(self.world.pods[:2], key=lambda pod: pod.score(use_cp_dist_score=self.use_cp_dist_score))
            current_score = best_pod.score(use_cp_dist_score=self.use_cp_dist_score)
            opp_current_score = max(pod.score(use_cp_dist_score=self.use_cp_dist_score) for pod in self.world.pods[2:])

        agent_solution = self._action_to_solution(action)

        # Dummy solution : straight line toward the next checkpoint
        if self.opp_solution_predict is None:
            opp_solution = Solution(
                move1=self.world.pods[2].to_dummy_move(speed=80.0),
                move2=self.world.pods[3].to_dummy_move(speed=80.0),
            )
        else:
            opp_action = self.opp_solution_predict(self._get_state(opponent_view=True))
            opp_solution = self._action_to_solution(opp_action)

        safe_state = self._get_state()
        try:
            self.world.play(agent_solution, opp_solution)
        except Exception as e:
            print('An error occurred during game generation!', e)
            return safe_state, 0.0, True, {}

        if self.world.player_won(1):
            episode_over = True
            reward = 0.0 if not self.use_raw_rewards else -10.0
        elif self.world.player_won(0):
            episode_over = True
            reward = 20.0 if not self.use_raw_rewards else 10.0
        else:
            episode_over = False
            if not self.use_raw_rewards:
                now_score = best_pod.score(use_cp_dist_score=self.use_cp_dist_score)
                opp_now_score = max(pod.score(use_cp_dist_score=self.use_cp_dist_score) for pod in self.world.pods[2:])
                reward = now_score - current_score - 0.1 * (opp_now_score - opp_current_score)
            else:
                reward = 0.0

        # assert self.reward_range[0] <= reward <= self.reward_range[1]
        return self._get_state(), reward, episode_over, {}

    def reset(self):
        self.world = World()
        return self._get_state()

    def render(self, mode='human'):
        if DISABLE_RENDERING:
            return None

        self.viewer, rendered = self.world.render(viewer=self.viewer, mode=mode)
        return rendered

    def switch_to_hard_env(self):
        self.use_cp_dist_score = False
        self.use_raw_rewards = True

    def enable_opponent(self, opp_solution_predict):
        self.opp_solution_predict = opp_solution_predict
