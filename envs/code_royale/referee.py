import gym
import numpy as np
from envs.code_royale.game import Game
from envs.code_royale.game import Move
from envs.code_royale.constants import *

MAX_NUM_SITES = 20  # TODO
MAX_NUM_UNITS = 20  # TODO


class Referee(gym.Env):

    reward_range = (-1, 1)

    def __init__(self):
        self.game = Game()

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32, shape=(2 + MAX_NUM_SITES + 1 + 1,))

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=(len(self._get_state()),))

    def reset(self):
        self.game = Game()
        return self._get_state()

    def _get_state(self):

        features = []
        for player in self.game.players:
            features += [player.queen.x, player.queen.y, player.queen.hp]

        for site in self.game.sites[:MAX_NUM_SITES]:
            features += [
                1,
                site.x,
                site.y,
                site.ignore1,
                site.ignore2,
                site.structure_type,
                site.player.id,
                site.player.param1,
                site.player.param2,
            ]

        for i in range(MAX_NUM_SITES - len(self.game.sites)):
            features += [0, 0, 0, 0, 0, 0, 0, 0, 0]

        for unit in self.game.units[:MAX_NUM_UNITS]:
            features += [
                1,
                unit.x,
                unit.y,
                unit.player.id,
                unit.unit_type,
                unit.health,
            ]

        for i in range(MAX_NUM_UNITS - len(self.game.units)):
            features += [0, 0, 0, 0, 0, 0]

        return np.array(features)

    def _action_to_move(self, action):
        target_x = action[0] * WORLD_WIDTH
        target_y = action[1] * WORLD_HEIGHT

        should_train_site = [
            act > 0.9 for act in action[2:2+MAX_NUM_SITES]
        ]

        if action[-2] <= 0.5:
            build_type = NO_STRUCTURE
        elif action[-2] <= 0.65:
            build_type = BARRACKS
        elif action[-2] <= 0.8:
            build_type = TOWER
        else:
            build_type = MINE

        site_id_to_build = int(action[-1] * MAX_NUM_SITES)
        return Move(
            target_x=target_x,
            target_y=target_y,
            should_train_site=should_train_site,
            build_type=build_type,
            site_id_to_build=site_id_to_build,
        )

    def step(self, action, compute_opponent_fn=None):

        if compute_opponent_fn is not None:
            opponent_move = self._action_to_move(compute_opponent_fn())
        else:
            opponent_move = Move(
                target_x=self.game.players[1].queen.x,
                target_y=self.game.players[1].queen.y,
                should_train_site=[False] * MAX_NUM_SITES,
                build_type=NO_STRUCTURE,
                site_id_to_build=-1
            )

        self.game.play(self._action_to_move(action), opponent_move)

        if self.game.winner == 0:
            reward = 1
        elif self.game.winner == 1:
            reward = -1
        else:
            reward = 0

        return self._get_state(), reward, self.game.is_game_over(), {}
