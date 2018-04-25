from collections import namedtuple
from envs.code_royale.players import Player


class Game:

    def __init__(self):
        self.player_1 = Player()
        self.player_2 = Player()


Move = namedtuple('Move', ['target_x', 'target_y', 'should_train_site', 'build_type', 'site_id_to_build'])
