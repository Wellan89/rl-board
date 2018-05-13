import random
import math

from envs.csb.circuit import Circuit
from envs.csb.pod import Pod


class World:

    __slots__ = ('pods', 'circuit', 'nblaps', 'turn')

    def __init__(self):
        self.circuit = Circuit()
        self.nblaps = 3
        self.turn = 0

        distance_to_center = 500.0 if random.random() < 0.5 else 1500.0
        cp0x = self.circuit.cp(0).x
        cp0y = self.circuit.cp(0).y
        angle = math.pi / 2.0 + math.atan2(self.circuit.cp(1).y - cp0y, self.circuit.cp(1).x - cp0x)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        self.pods = [
            Pod(0, cp0x + cos_angle * distance_to_center,
                cp0y + sin_angle * distance_to_center, self),
            Pod(1, cp0x - cos_angle * distance_to_center,
                cp0y - sin_angle * distance_to_center, self),
            Pod(2, cp0x + cos_angle * (2000.0 - distance_to_center),
                cp0y + sin_angle * (2000.0 - distance_to_center), self),
            Pod(3, cp0x - cos_angle * (2000.0 - distance_to_center),
                cp0y - sin_angle * (2000.0 - distance_to_center), self),
        ]
        for pod in self.pods:
            pod.angle = (angle - math.pi / 2.0) * 180.0 / math.pi

    def play(self, s1, s2):
        self.turn += 1

        for pod in self.pods:
            if pod.shield > 0:
                pod.shield -= 1

        self.pods[0].apply_move(s1[:3])
        self.pods[1].apply_move(s1[3:])
        self.pods[2].apply_move(s2[:3])
        self.pods[3].apply_move(s2[3:])

        t = 0.0
        lasta = set()
        lastb = set()
        while True:
            firstCol = None
            for i, pod_i in enumerate(self.pods):
                for j, col_b in enumerate(self.pods):
                    if j > i and (id(pod_i) not in lasta or id(col_b) not in lastb):
                        col_t = pod_i.collision(col_b)
                        if col_t is not None and col_t + t < 1.0 and (not firstCol or col_t < firstCol[2]):
                            firstCol = (pod_i, col_b, col_t)

                col_b = self.circuit.cps[pod_i.ncpid]
                col_t = pod_i.collision(col_b)
                if col_t is not None and col_t + t < 1.0 and (not firstCol or col_t < firstCol[2]):
                    firstCol = (pod_i, col_b, col_t)

            if firstCol:
                col_a, col_b, col_t = firstCol
                if col_t > 0.0:
                    lasta.clear()
                    lastb.clear()
                lasta.add(id(col_a))
                lastb.add(id(col_b))

                for pod in self.pods:
                    pod.move(col_t)
                t += col_t

                col_b.bounce(col_a)
            else:
                for pod in self.pods:
                    pod.move(1.0 - t)
                break

        for pod in self.pods:
            pod.end()

    def player_won(self, player):
        # assert player == 0 or player == 1

        # Race finished
        if self.pods[player*2].lap == self.nblaps or self.pods[player*2+1].lap == self.nblaps:
            return True

        # Opponent timeout
        if self.pods[(1-player)*2].timeout < 0 and self.pods[(1-player)*2+1].timeout < 0:
            return True

        return False

    def best_pod(self, player):
        if self.pods[player*2].score() > self.pods[player*2+1].score():
            return self.pods[player*2]
        return self.pods[player*2+1]

    def second_pod(self, player):
        if self.pods[player*2].score() > self.pods[player*2+1].score():
            return self.pods[player*2+1]
        return self.pods[player*2]

    def dummy_opp_solution(self):
        return self.pods[2].to_dummy_move(speed=80.0) + self.pods[3].to_dummy_move(speed=80.0)

    def compute_state(self, opponent_view=False):
        if opponent_view:
            pods = self.pods[2:] + self.pods[:2]
        else:
            pods = self.pods

        features = [float(self.nblaps), float(self.circuit.nbcp())]
        for pod in pods:
            features += [
                pod.x / 5000.0,
                pod.y / 5000.0,
                pod.vx / 5000.0,
                pod.vy / 5000.0,
                pod.angle / 360.0,
                float(pod.boost_available),
                pod.timeout / 100.0,
                pod.shield / 4.0,
                float(pod.lap),
                float(pod.ncpid),
            ]
            for i in range(3):
                cp = pod.next_checkpoint(i)
                features += [
                    cp.x / 5000.0,
                    cp.y / 5000.0,
                ]
        return features
