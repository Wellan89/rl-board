import math


def LIN(x, x1, y1, x2, y2):
    return y1 + (y2-y1)*(x-x1)/(x2-x1)


def correct_angle_rad(angle):
    while angle <= math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


MAX_THRUST = 200
NBPOD = 4
TIMEOUT = 100
