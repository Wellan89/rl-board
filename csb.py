import base64
import math
import sys
import time

import numpy as np


MODEL_DATA = {}


def LIN(x, x1, y1, x2, y2):
    return y1 + (y2-y1)*(x-x1)/(x2-x1)


MAX_THRUST = 200
NBPOD = 4
TIMEOUT = 100


class Model:
    def __init__(self, weights):
        self.weights = weights

    @classmethod
    def from_data(cls, model_data):
        weights = {}
        for var, (shape, data) in model_data.items():
            weights[var] = np.reshape(
                np.fromstring(base64.decodebytes(data.encode()), dtype=np.float16).astype(np.float32), shape
            )
        return cls(weights=weights)

    def predict(self, state):
        layer = np.array([state], dtype=np.float32)
        layer = np.tanh(np.matmul(layer, self.weights['dense0/W']) + self.weights['dense0/b'])
        layer = np.tanh(np.matmul(layer, self.weights['dense1/W']) + self.weights['dense1/b'])

        log_eps = math.log(1e-6)
        alpha = np.matmul(layer, self.weights['alpha/W']) + self.weights['alpha/b']
        alpha = np.log(np.exp(np.clip(alpha, log_eps, -log_eps)) + 1.0) + 1.0

        beta = np.matmul(layer, self.weights['beta/W']) + self.weights['beta/b']
        beta = np.log(np.exp(np.clip(beta, log_eps, -log_eps)) + 1.0) + 1.0

        alpha_beta = np.maximum(alpha + beta, 1e-6)
        definite = beta / alpha_beta

        return definite[0]

    def compute_action(self, game_state):
        action = self.predict(game_state.extract_state())
        return Action(game_state, action)


class Action:
    def __init__(self, game_state, action):
        self.game_state = game_state
        self.action = list(action)

    def output(self):
        for i in range(2):
            self.game_state.pods[i].output(self.action[3*i:3*(i+1)])


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Pod:
    def __init__(self):
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None
        self.angle = None
        self.next_check_point_id = None
        self.boost_available = True
        self.shield = 0
        self.lap = 0
        self.timeout = TIMEOUT

    def read_turn(self):
        x, y, vx, vy, angle, next_check_point_id = map(int, input().split())

        if (self.next_check_point_id != next_check_point_id):
            self.timeout = TIMEOUT
            if (next_check_point_id == 1):
                self.lap += 1

        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.angle = angle
        self.next_check_point_id = next_check_point_id

        if self.shield > 0:
            self.shield -= 1

    def next_checkpoint(self, game_state, number_next):
        target_cpid = (self.next_check_point_id + number_next) % len(game_state.checkpoints)
        return game_state.checkpoints[target_cpid]

    def nb_checked(self, game_state):
        last_cp = self.next_check_point_id - 1
        if last_cp == -1:
            last_cp = len(game_state.checkpoints) - 1
        return self.lap * len(game_state.checkpoints) + last_cp

    def get_new_angle(self, gene):
        res = self.angle
        res += LIN(gene, 0.0, -18.0, 1.0, 18.0)

        if res >= 360.0:
            res -= 360.0
        elif res < 0.0:
            res += 360.0

        return res

    def get_new_power(self, gene):
        return LIN(gene, 0.0, 0, 1.0, MAX_THRUST)

    def output(self, move):
        a = self.get_new_angle(move[0]) * math.pi / 180.0
        px = self.x + math.cos(a) * 1000000.0
        py = self.y + math.sin(a) * 1000000.0
        power = self.get_new_power(move[1])
        if move[2] < 0.05 and self.boost_available:
            self.boost_available = False
            print('{:.0f} {:.0f} BOOST'.format(px, py))
        elif move[2] > 0.95:
            self.shield = 4
            print('{:.0f} {:.0f} SHIELD'.format(px, py))
        else:
            print('{:.0f} {:.0f} {:.0f}'.format(px, py, power))
        self.timeout -= 1


class GameState:
    def __init__(self, laps, checkpoints):
        self.laps = laps
        self.checkpoints = checkpoints
        self.pods = [Pod() for _ in range (4)]

    @classmethod
    def read_initial(cls):
        laps = int(input())
        checkpoint_count = int(input())
        checkpoints = [Point(*[int(j) for j in input().split()]) for i in range(checkpoint_count)]
        return cls(laps, checkpoints)

    def read_turn(self):
        for pod in self.pods:
            pod.read_turn()

    def extract_state(self):
        features = [len(self.checkpoints) * self.laps]
        for pod in self.pods:
            features += [
                pod.x / 1000,
                pod.y / 1000,
                pod.vx / 1000,
                pod.vy / 1000,
                pod.angle / 360,
                float(pod.boost_available),
                pod.timeout / TIMEOUT,
                pod.shield / 4,
                pod.nb_checked(self),
            ]
            for i in range(3):
                cp = pod.next_checkpoint(self, i)
                features += [
                    cp.x / 1000,
                    cp.y / 1000,
                ]
        for i in range(6):
            cp = self.checkpoints[i] if i < len(self.checkpoints) else Point(0.0, 0.0)
            features += [
                cp.x / 1000,
                cp.y / 1000,
            ]
        return features


def main():
    t0 = time.time()
    ai = Model.from_data(MODEL_DATA)
    print('Took {:.2f}ms to load the model'.format((time.time() - t0) * 1000), file=sys.stderr)

    game_state = GameState.read_initial()
    while True:
        game_state.read_turn()
        t0 = time.time()
        action = ai.compute_action(game_state)
        print('Took {:.2f}ms to predict the action'.format((time.time() - t0) * 1000), file=sys.stderr)
        action.output()


if __name__ == "__main__":
    main()
