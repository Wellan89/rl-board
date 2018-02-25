from collision import Collision
from circuit import Circuit


class World:

    def __init__(self):
        self.reset()

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
        previousCollision = False
        lasta = None
        lastb = None

        while t < 1.0:
            firstCol = Collision(None, None, -1.0)
            foundCol = False
            for i in range(len(self.pods)):
                for j in range(i+1, len(self.pods)):
                    col = self.pods[i].collision(self.pods[j])
                    if col is not None and col.t + t < 1.0 and (foundCol is False or col.t < firstCol.t):
                        firstCol = col
                        foundCol = True

                col = self.pods[i].collision(self.circuit.cps[self.pods[i].ncpid])
                if col is not None and col.t + t < 1.0 and (foundCol is False or col.t < firstCol.t):
                    firstCol = col
                    foundCol = True

            if foundCol is False or (
                previousCollision and firstCol.t == 0.0 and firstCol.a == lasta and firstCol.b == lastb
            ):
                for pod in self.pods:
                    pod.move(1.0-t)
                t = 1.0
            else:
                previousCollision = True
                lasta = firstCol.a
                lastb = firstCol.b
                for pod in self.pods:
                    pod.move(firstCol.t)

                firstCol.b.bounce(firstCol.a)
                t += firstCol.t

        for pod in self.pods:
            pod.end()

    def reset(self):
        self.circuit = Circuit()
        self.turn = 0
        self.pods = [None] * 4  # TODO : Pod placement (waiting for CG response) !!!!!
        self.nblaps = 42  # TODO : Number of laps (Waiting for CG response) !!!!

    def player_won(self, player):
        assert player == 0 or player == 1

        # Race finished
        if self.pods[player*2].lap == self.nblaps or self.pods[player*2+1].lap == self.nblaps:
            return True

        # Opponent timeout
        return self.pods[(1-player)*2].timeout < 0 or self.pods[(1-player)*2+1].timeout < 0

    def best_pod(self, player):
        if self.pods[player*2].score() > self.pods[player*2+1].score():
            return self.pods[player*2]
        return self.pods[player*2+1]

    def second_pod(self, player):
        if self.pods[player*2].score() > self.pods[player*2+1].score():
            return self.pods[player*2+1]
        return self.pods[player*2]
