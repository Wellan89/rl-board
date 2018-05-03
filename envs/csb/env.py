import os
import gym
import numpy as np

from envs.csb import observation
from envs.csb.world import World
from envs.csb.solution import Solution
from envs.csb.move import Move

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

    def _compute_score(self):
        return 0.5 * sum(pod.score(use_cp_dist_score=True) for pod in self.world.pods[:2])

    def step(self, action):
        # assert (len(action),) == self.action_space.shape
        # assert all(self.action_space.low <= v <= self.action_space.high for v in action)
        action = np.clip(action, 0.0, 1.0)
        agent_solution = self._action_to_solution(action)

        last_score = self._compute_score()

        opp_action = None
        if self.opp_solution_predict is not None:
            opp_action = self.opp_solution_predict(self._get_state(opponent_view=True), self.is_new_episode)
        self.is_new_episode = False

        # Dummy solution : straight line toward the next checkpoint
        if opp_action is None:
            opp_solution = Solution(
                move1=self.world.pods[2].to_dummy_move(speed=80.0),
                move2=self.world.pods[3].to_dummy_move(speed=80.0),
            )
        else:
            opp_solution = self._action_to_solution(opp_action)

        safe_state = self._get_state()
        try:
            self.world.play(agent_solution, opp_solution)
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
            easy_reward = self._compute_score() - last_score
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
