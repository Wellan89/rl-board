from envs.csb.unit import Unit
from envs.csb.util import TIMEOUT


class Checkpoint(Unit):

    def __init__(self, id, x, y):
        super().__init__(id, x, y, 200, 0, 0)

    def bounce(self, pod):
        pod.incrcpid()
        pod.timeout = TIMEOUT
