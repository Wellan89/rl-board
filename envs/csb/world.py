import random
import math
from envs.csb.collision import Collision
from envs.csb.circuit import Circuit
from envs.csb.pod import Pod


class World:

    __slots__ = ('pods', 'circuit', 'nblaps', 'turn')

    def __init__(self):
        self.circuit = Circuit()
        self.nblaps = 3
        self.turn = 0

        distance_to_center = random.choice([500, 1500])
        cp0x = self.circuit.cp(0).x
        cp0y = self.circuit.cp(0).y
        angle = math.pi / 2 + math.atan2(self.circuit.cp(1).y - cp0y, self.circuit.cp(1).x - cp0x)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        self.pods = [
            Pod(0, cp0x + cos_angle * distance_to_center,
                cp0y + sin_angle * distance_to_center, self),
            Pod(1, cp0x - cos_angle * distance_to_center,
                cp0y - sin_angle * distance_to_center, self),
            Pod(2, cp0x + cos_angle * (2000 - distance_to_center),
                cp0y + sin_angle * (2000 - distance_to_center), self),
            Pod(3, cp0x - cos_angle * (2000 - distance_to_center),
                cp0y - sin_angle * (2000 - distance_to_center), self),
        ]
        for pod in self.pods:
            pod.angle = angle - math.pi / 2

    def play(self, s1, s2):

        self.turn += 1

        for pod in self.pods:
            if pod.shield > 0:
                pod.shield -= 1

        self.pods[0].apply_move(s1.move1)
        self.pods[1].apply_move(s1.move2)
        self.pods[2].apply_move(s2.move1)
        self.pods[3].apply_move(s2.move2)

        t = 0.0
        previousCollision = False
        lasta = None
        lastb = None

        while t < 1.0:
            firstCol = Collision(None, None, -1.0)
            foundCol = False
            for i in range(len(self.pods)):
                for j in range(i+1, len(self.pods)):
                    col = self.pods[i].collision(self.pods[j])
                    if col is not None and col.t + t < 1.0 and (foundCol is False or col.t < firstCol.t):
                        firstCol = col
                        foundCol = True

                col = self.pods[i].collision(self.circuit.cps[self.pods[i].ncpid])
                if col is not None and col.t + t < 1.0 and (foundCol is False or col.t < firstCol.t):
                    firstCol = col
                    foundCol = True

            if foundCol is False or (
                previousCollision and firstCol.t == 0.0 and firstCol.a == lasta and firstCol.b == lastb
            ):
                for pod in self.pods:
                    pod.move(1.0-t)
                t = 1.0
            else:
                previousCollision = True
                lasta = firstCol.a
                lastb = firstCol.b
                for pod in self.pods:
                    pod.move(firstCol.t)

                firstCol.b.bounce(firstCol.a)
                t += firstCol.t

        for pod in self.pods:
            pod.end()

    def player_won(self, player, enable_timeout=True):
        assert player == 0 or player == 1

        # Race finished
        if self.pods[player*2].lap == self.nblaps or self.pods[player*2+1].lap == self.nblaps:
            return True

        # Opponent timeout
        if enable_timeout and self.pods[(1-player)*2].timeout < 0 and self.pods[(1-player)*2+1].timeout < 0:
            return True

        if self.turn > 900:
            print('Warning: Abnormally long game! The opponent wins')
            return player == 1

        return False

    def best_pod(self, player):
        if self.pods[player*2].score() > self.pods[player*2+1].score():
            return self.pods[player*2]
        return self.pods[player*2+1]

    def second_pod(self, player):
        if self.pods[player*2].score() > self.pods[player*2+1].score():
            return self.pods[player*2+1]
        return self.pods[player*2]
