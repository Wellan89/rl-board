import math

from envs.csb.unit import Unit
from envs.csb.point import Point
from envs.csb.util import MAX_THRUST, TIMEOUT


class Pod(Unit):

    __slots__ = ('angle', 'ncpid', 'timeout', 'shield', 'lap', 'world', 'boost_available')

    def __init__(self, id, x, y, world):
        super().__init__(id, x, y, 400, 0, 0)
        self.angle = 0
        self.ncpid = 1
        self.timeout = TIMEOUT
        self.shield = 0
        self.lap = 0
        self.world = world
        self.boost_available = True

    def incrcpid(self):
        self.ncpid += 1
        if self.ncpid == 1:
            self.lap += 1
        if self.ncpid >= self.world.circuit.nbcp():
            self.ncpid = 0

    def nbChecked(self):
        lastCP = self.ncpid-1
        if lastCP == -1:
            lastCP = self.world.circuit.nbcp() - 1
        return self.lap * self.world.circuit.nbcp() + lastCP

    def score(self, use_cp_dist_score=False):
        if not use_cp_dist_score:
            return self.nbChecked()

        current_cp = self.world.circuit.cp((self.ncpid - 1) % self.world.circuit.nbcp())
        next_cp = self.world.circuit.cp(self.ncpid)
        distance_cp_to_ncp = current_cp.distance(next_cp)
        cp_dist_score = (distance_cp_to_ncp - self.distance(next_cp)) / distance_cp_to_ncp
        return self.nbChecked() + cp_dist_score

    def block_score(self, opp_run_pod):
        return -self.distance(opp_run_pod.next_checkpoint()) / 5000.0

    def diffAngle(self, p):
        a = math.atan2(p.y - self.y, p.x - self.x) * 180.0 / math.pi
        right = a - self.angle if self.angle <= a else 360.0 - self.angle + a
        left = self.angle - a if self.angle >= a else self.angle + 360.0 - a
        return right if right < left else -left

    def rotate(self, p):
        a = self.diffAngle(p)
        if a > 18.0:
            a = 18.0
        elif a < -18.0:
            a = -18.0

        self.angle += a
        if self.angle >= 360.0:
            self.angle -= 360.0
        elif self.angle < 0.0:
            self.angle += 360.0

    def boost(self, thrust):
        # assert self.shield == 0
        ra = self.angle * math.pi / 180.0
        self.vx += math.cos(ra) * thrust
        self.vy += math.sin(ra) * thrust

    def move(self, t):
        self.x += self.vx*t
        self.y += self.vy*t

    def end(self):
        self.x = int(self.x + 0.5)
        self.y = int(self.y + 0.5)
        self.vx = int(self.vx * 0.85)
        self.vy = int(self.vy * 0.85)
        self.timeout -= 1

    def bounce(self, unit):
        tm = 10.0 if self.shield == 4 else 1.0
        em = 10.0 if unit.shield == 4 else 1.0
        mcoeff = (tm + em) / (tm * em)
        nx = self.x - unit.x
        ny = self.y - unit.y
        nxnysquare = nx*nx + ny*ny
        dvx = self.vx - unit.vx
        dvy = self.vy - unit.vy
        product = nx*dvx + ny*dvy
        fx = (nx * product) / (nxnysquare * mcoeff)  # Can raise ZeroDivisionError: float division by zero
        fy = (ny * product) / (nxnysquare * mcoeff)

        self.vx -= fx / tm
        self.vy -= fy / tm
        unit.vx += fx / em
        unit.vy += fy / em

        impulse = math.sqrt(fx*fx + fy*fy)
        if impulse < 120.0:
            fx = fx * 120.0 / impulse
            fy = fy * 120.0 / impulse

        self.vx -= fx / tm
        self.vy -= fy / tm
        unit.vx += fx / em
        unit.vy += fy / em

    def next_checkpoint(self, number_next=0):
        return self.world.circuit.cp((self.ncpid + number_next) % self.world.circuit.nbcp())

    def get_new_angle(self, gene):
        res = self.angle + gene * 36.0 - 18.0
        if res >= 360.0:
            res -= 360.0
        elif res < 0.0:
            res += 360.0
        return res

    def get_new_power(self, gene):
        return gene * MAX_THRUST

    def apply_move(self, move):
        self.angle = self.get_new_angle(move[0])
        if move[2] <= 0.2 and self.boost_available:
            if self.shield == 0:
                self.boost_available = False
                self.boost(650.0)
        elif move[2] >= 0.8:
            self.shield = 4
        elif self.shield == 0:
            self.boost(self.get_new_power(move[1]))

    def to_dummy_move(self, speed):
        return [
            (self.diffAngle(self.world.circuit.cp(self.ncpid)) + 18.0) / 36.0,
            speed / MAX_THRUST,
            0.5,
        ]

    def genes_from_vincent_command(self, command):
        return [
            (self.diffAngle(Point(command.target.x, command.target.y)) + 18.0) / 36.0,
            command.thrust / MAX_THRUST,
            1.0 if command.shield else 0.5,
        ]
