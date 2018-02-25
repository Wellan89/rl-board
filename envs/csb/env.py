import gym
from world import World
from solution import Solution
from move import Move
from observation import Observation


class CsbEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.world = World()

    def _step(self, action):

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

        return Observation(self.world), reward, episode_over, self._get_debug()

    def _check_action_format(self, action):
        assert len(action) == 2
        for pod_action in action:
            assert len(pod_action) == 3
            for gene in pod_action:
                assert 0 <= gene <= 1

    def _reset(self):
        self.world.reset()

    def _render(self, mode='human', close=False):
        pass

    def _get_debug(self):
        return {
            '42': 42
        }
