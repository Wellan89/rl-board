import random
from envs.csb.point import Point
from envs.csb.checkpoint import Checkpoint


CHECKPOINT_MAX_DEVIATION = 30

BASE_CONFIGURATIONS = [
    [Point(10540, 5980), Point(3580, 5180), Point(13580, 7600), Point(12460, 1350)],
    [Point(13840, 5080), Point(10680, 2280), Point(8700, 7460), Point(7200, 2160), Point(3600, 5280)],
    [Point(7350, 4940), Point(3320, 7230), Point(14580, 7700), Point(10560, 5060), Point(13100, 2320), Point(4560, 2180)],
    [Point(11480, 6080), Point(9100, 1840), Point(5010, 5260)],
    [Point(3450, 7220), Point(9420, 7240), Point(5970, 4240), Point(14660, 1410)],
    [Point(8000, 7900), Point(13300, 5540), Point(9560, 1400), Point(3640, 4420)],
    [Point(13500, 2340), Point(12940, 7220), Point(5640, 2580), Point(4100, 7420)],
    [Point(6320, 4290), Point(7800, 860), Point(7660, 5970), Point(3140, 7540), Point(9520, 4380), Point(14520, 7780)],
    [Point(13920, 1940), Point(8020, 3260), Point(2670, 7020), Point(10040, 5970)],
    [Point(6000, 5360), Point(11300, 2820), Point(7500, 6940)],
    [Point(13040, 1900), Point(6560, 7840), Point(7480, 1360), Point(12700, 7100), Point(4060, 4660)],
    [Point(6280, 7760), Point(14100, 7760), Point(13880, 1220), Point(10240, 4920), Point(6100, 2200), Point(3020, 5190)],
    [Point(11203, 5425), Point(7259, 6656), Point(5425, 2838), Point(10323, 3366)],
]

MAX_NB_CHECKPOINTS = max(map(len, BASE_CONFIGURATIONS))


class Circuit:

    def __init__(self):
        self.cps = [
            Checkpoint(
                i,
                point.x + random.randint(-CHECKPOINT_MAX_DEVIATION, CHECKPOINT_MAX_DEVIATION),
                point.y + random.randint(-CHECKPOINT_MAX_DEVIATION, CHECKPOINT_MAX_DEVIATION),
            )
            for i, point in enumerate(random.choice(BASE_CONFIGURATIONS))
        ]

    def nbcp(self):
        return len(self.cps)

    def cp(self, cpid):
        return self.cps[cpid]
