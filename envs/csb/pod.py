import math

from envs.csb.move import Move
from envs.csb.unit import Unit
from envs.csb.point import Point
from envs.csb.util import LIN, MAX_THRUST, TIMEOUT


class Pod(Unit):

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

    def getAngle(self, p):
        d = self.distance(p)
        if d == 0:
            return 0
        dx = (p.x - self.x) / d
        dy = (p.y - self.y) / d
        a = math.acos(dx) * 180.0 / math.pi
        if dy < 0:
            a = 360.0-a
        return a

    def diffAngle(self, p):
        a = self.getAngle(p)
        right = a - self.angle if self.angle <= a else 360.0 - self.angle + a
        left = self.angle - a if self.angle >= a else self.angle + 360.0 - a
        if right < left:
            return right
        return -left

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
        if self.shield > 0:
            return
        ra = self.angle * math.pi / 180.0
        self.vx += math.cos(ra) * thrust
        self.vy += math.sin(ra) * thrust

    def move(self, t):
        self.x += self.vx*t
        self.y += self.vy*t

    def end(self):
        self.x = round(self.x)
        self.y = round(self.y)
        self.vx = int(self.vx * 0.85)
        self.vy = int(self.vy * 0.85)
        self.timeout -= 1

    def bounce(self, unit):
        tm = 10 if self.shield == 4 else 1
        em = 10 if unit.shield == 4 else 1
        mcoeff = (tm + em) / (tm * em)
        nx = self.x - unit.x
        ny = self.y - unit.y
        nxnysquare = nx*nx + ny*ny
        dvx = self.vx - unit.vx
        dvy = self.vy - unit.vy
        product = nx*dvx + ny*dvy
        fx = (nx * product) / (nxnysquare * mcoeff)
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

    def get_new_angle(self, gene):
        res = self.angle
        res += LIN(gene, 0.0, -18.0, 1.0, 18.0)

        if res >= 360.0:
            res -= 360.0
        elif res < 0.0:
            res += 360.0

        return res

    def get_new_power(self, gene):
        return LIN(gene, 0.0, 0.0, 1.0, MAX_THRUST)

    def apply_move(self, move):
        self.angle = self.get_new_angle(move.g1)
        if move.g3 < 0.05 and self.boost_available:
            if self.shield == 0:
                self.boost_available = False
                self.boost(650)
        elif move.g3 > 0.95:
            self.shield = 4
        else:
            if self.shield == 0:
                self.boost(self.get_new_power(move.g2))

    def output(self, move):
        a = self.get_new_angle(move.g1) * math.pi / 180.0
        px = self.x + math.cos(a) * 1000000.0
        py = self.y + math.sin(a) * 1000000.0
        power = self.get_new_power(move.g2)
        if move.g3 < 0.05 and self.boost_available:
            print('{} {} BOOST'.format(px, py))
        elif move.g3 > 0.95:
            print('{} {} SHIELD'.format(px, py))
        else:
            print('{} {} {}'.format(px, py, round(power)))

    def next_checkpoint(self, world, number_next=0):
        target_cpid = (self.ncpid + number_next) % world.circuit.nbcp()
        return world.circuit.cp(target_cpid)

    def to_dummy_move(self, speed):
        if speed == 0.0:
            return Move(0.5, 0.0, 0.5)

        next_cp = self.world.circuit.cp(self.ncpid)
        return Move(
            g1=min(max(LIN(self.diffAngle(next_cp), -18.0, 0.0, 18.0, 1.0), 0.0), 1.0),
            g2=speed,
            g3=0.5
        )

    def genes_from_vincent_command(self, command):
        return Move(
            g1=max(min(LIN(self.diffAngle(Point(command.target.x, command.target.y)), -18.0, 0.0, 18.0, 1.0), 1.0), 0.0),
            g2=max(min(LIN(command.thrust, 0.0, 0.0, MAX_THRUST, 1.0), 1.0), 0.0),
            g3=1.0 if command.shield else 0.5,
        )
