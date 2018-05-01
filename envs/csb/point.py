import math


class Point:

    __slots__ = ('x', 'y')

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance2(self, point):
        return (self.x - point.x)**2 + (self.y - point.y)**2

    def distance(self, point):
        return math.sqrt(self.distance2(point))

    def closest(self, a, b):
        da = b.y - a.y
        db = a.x - b.x
        c1 = da * a.x + db * a.y
        c2 = -db * self.x + da * self.y
        det = da * da + db * db
        if det:
            cx = (da*c1 - db*c2) / det
            cy = (da*c2 + db*c1) / det
        else:
            cx = self.x
            cy = self.y
        return Point(cx, cy)
