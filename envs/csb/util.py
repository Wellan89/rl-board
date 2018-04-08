def LIN(x, x1, y1, x2, y2):
    return y1 + (y2-y1)*(x-x1)/(x2-x1)


def CLAMP(x, x_min, x_max):
    return min(max(x, x_min), x_max)


MAX_THRUST = 200.0
NBPOD = 4
TIMEOUT = 100
