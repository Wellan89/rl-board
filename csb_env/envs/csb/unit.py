from point import Point
from collision import Collision
import math


class Unit(Point):

    def __init__(self, id, x, y, r, vx, vy):
        super().__init__(x, y)
        self.id = id
        self.r = r
        self.vx = vx
        self.vy = vy

    # Retourne la prochaine collision avec 'unit'
    def collision(self, unit):
        dist = self.distance2(unit)
        somme_rayon_2 = (self.r + unit.r) * (self.r + unit.r)
        if dist < somme_rayon_2:
            return Collision(self, unit, 0.0)

        if self.vx == unit.vx and self.vy == unit.vy:
            return None

        x = self.x - unit.x
        y = self.y - unit.y
        myp = Point(x, y)
        vx = self.vx - unit.vx
        vy = self.vy - unit.vy
        up = Point(0.0, 0.0)
        p = up.closest(myp, Point(x+vx, y+vy))

        pdist = up.distance2(p)
        mypdist = myp.distance2(p)

        if pdist >= somme_rayon_2:
            return None

        length = math.sqrt(vx*vx + vy*vy)
        backdist = math.sqrt(somme_rayon_2 - pdist)
        p.x = p.x - backdist * (vx / length)
        p.y = p.y - backdist * (vy / length)

        if myp.distance2(p) > mypdist:
            return None

        pdist = p.distance(myp)

        if pdist >= length:
            return None

        return Collision(self, unit, pdist / length)

    def bounce(self, unit):
        raise NotImplementedError('Abstract method')
