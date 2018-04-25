from envs.code_royale.constants import *


class Unit:
    def __init__(self, player, radius, speed, hp):
        self.player = player
        self.radius = radius
        self.speed = speed
        self.hp = hp


class Queen(Unit):

    def __init__(self, player, hp):
        super().__init__(player, radius=QUEEN_RADIUS, speed=QUEEN_SPEED, hp=hp)

    def build(self, site):
        pass


class Creep(Unit):
    pass


class Knight(Creep):
    pass


class Archer(Creep):
    pass


class Giant(Creep):
    pass
