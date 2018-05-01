import random
import math

from envs.csb.circuit import Circuit
from envs.csb.pod import Pod
from envs.csb.vincent_algo import Pos


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
            pod.angle = angle - math.pi / 2.0

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

    def render(self, viewer=None, mode='human'):
        from gym.envs.classic_control import rendering

        VIEWPORT_W = 710
        VIEWPORT_H = 400

        if viewer is None:
            viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

        def _pos_to_screen(_p):
            return _p.x * VIEWPORT_W / 16000, _p.y * VIEWPORT_H / 9000

        def _radius_to_screen(_r):
            return _r * VIEWPORT_W / 16000

        cp_radius = _radius_to_screen(self.circuit.cps[0].r + self.pods[0].r)
        for cp in self.circuit.cps:
            color_ratio = cp.id / (len(self.circuit.cps) - 1)
            color = (color_ratio * 0.8, color_ratio * 0.8, 0.2 + color_ratio * 0.8)
            viewer.draw_circle(color=color, radius=cp_radius).add_attr(
                rendering.Transform(translation=_pos_to_screen(cp))
            )

        pod_radius = _radius_to_screen(self.pods[0].r)
        for pod in self.pods:
            color = (float(pod.id >= 2), float(pod.id < 2), 0.0)
            if pod.shield > 0:
                viewer.draw_circle(color=(0.0, 0.0, 0.6),
                                   radius=pod_radius + _radius_to_screen(20 * pod.shield)).add_attr(
                    rendering.Transform(translation=_pos_to_screen(pod))
                )
            viewer.draw_circle(color=color, radius=pod_radius).add_attr(
                rendering.Transform(translation=_pos_to_screen(pod))
            )
            if pod.boost_available:
                viewer.draw_circle(color=(0.0, 0.0, 0.0),
                                   radius=_radius_to_screen(80)).add_attr(
                    rendering.Transform(translation=_pos_to_screen(pod))
                )
            viewer.draw_line(
                color=color,
                start=_pos_to_screen(pod),
                end=_pos_to_screen(pod.next_checkpoint()),
            )
            pod_angle_rad = pod.angle * math.pi / 180.0
            viewer.draw_line(
                color=(0.0, 0.0, 0.0),
                start=_pos_to_screen(pod),
                end=_pos_to_screen(Pos(pod.x + 300.0 * math.cos(pod_angle_rad),
                                       pod.y + 300.0 * math.sin(pod_angle_rad))),
            )

        return viewer, viewer.render(return_rgb_array=(mode == 'rgb_array'))
