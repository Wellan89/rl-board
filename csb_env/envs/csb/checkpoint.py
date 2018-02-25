from unit import Unit


class Checkpoint(Unit):

    def __init__(self, id, x, y):
        super().__init__(id, x, y, 200, 0, 0)

    def bounce(self, pod):
        pod.incrcpid()
        pod.timeout = 100
