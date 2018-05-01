import base64
import math
import sys
import time

import numpy as np


MODEL_DATA = {}


IS_CODINGAME = False
pod_msgs = ['', '']


def _matmul(a, b):
    return np.einsum('i,ij->j', a, b)


class Model:
    def __init__(self, weights, deterministic=True):
        assert weights
        self.weights = weights
        self.deterministic = deterministic

    @classmethod
    def from_data(cls, model_data):
        weights = {
            var: np.reshape(
                np.fromstring(base64.decodebytes(data.encode()), dtype=dtype).astype(np.float64), shape
            ) for var, (shape, dtype, data) in model_data.items()
        }
        return cls(weights=weights)

    def predict(self, state):
        obz = np.array(state, dtype=np.float64)
        state_mean = self.weights['pi/obfilter/runningsum:0'] / self.weights['pi/obfilter/count:0']
        state_std = np.sqrt(np.maximum(
            (self.weights['pi/obfilter/runningsumsq:0'] / self.weights['pi/obfilter/count:0']) - np.square(state_mean),
            1e-2
        ))
        obz = np.clip((obz - state_mean) / state_std, -5.0, 5.0)

        action = np.tanh(_matmul(obz, self.weights['pi/pol/fc1/kernel:0']) + self.weights['pi/pol/fc1/bias:0'])
        action = np.tanh(_matmul(action, self.weights['pi/pol/fc2/kernel:0']) + self.weights['pi/pol/fc2/bias:0'])
        action = _matmul(action, self.weights['pi/pol/final/kernel:0']) + self.weights['pi/pol/final/bias:0']

        if IS_CODINGAME:
            vf = np.tanh(_matmul(obz, self.weights['pi/vf/fc1/kernel:0']) + self.weights['pi/vf/fc1/bias:0'])
            vf = np.tanh(_matmul(vf, self.weights['pi/vf/fc2/kernel:0']) + self.weights['pi/vf/fc2/bias:0'])
            vf = float(_matmul(vf, self.weights['pi/vf/final/kernel:0']) + self.weights['pi/vf/final/bias:0'])
            print(action, file=sys.stderr)
            print(np.exp(self.weights['pi/pol/logstd:0'][0]), file=sys.stderr)
            print('{:.2f}'.format(vf), file=sys.stderr)
            pod_msgs[:] = [' {:.2f}'.format(vf)] * 2

        if not self.deterministic:
            action = np.random.normal(action, np.exp(self.weights['pi/pol/logstd:0'][0]))

        return np.clip(action, 0.0, 1.0)

    def compute_action(self, game_state):
        action = self.predict(game_state.extract_state())
        return Action(game_state, action)


class Action:
    def __init__(self, game_state, action):
        self.game_state = game_state
        self.action = list(action)

    def output(self):
        for i in [1, 0]:
            self.game_state.pods[i].output(self.action[3*i:3*(i+1)])


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Pod:
    def __init__(self, pod_id):
        self.id = pod_id
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None
        self.angle = None
        self.next_check_point_id = 1
        self.boost_available = True
        self.shield = 0
        self.lap = 0
        self.timeout = 100

    def read_turn(self):
        x, y, vx, vy, angle, next_check_point_id = map(int, input().split())

        if self.next_check_point_id != next_check_point_id:
            self.timeout = 100
            if next_check_point_id == 0:
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
        res = self.angle + gene * 36.0 - 18.0
        if res >= 360.0:
            res -= 360.0
        elif res < 0.0:
            res += 360.0
        return res

    def output(self, move):
        a = self.get_new_angle(move[0]) * math.pi / 180.0
        px = self.x + math.cos(a) * 1000000.0
        py = self.y + math.sin(a) * 1000000.0
        power = move[1] * 200.0
        if move[2] <= 0.2 and self.boost_available:
            self.boost_available = False
            print('{:.0f} {:.0f} BOOST{}'.format(px, py, pod_msgs[self.id]))
        elif move[2] >= 0.8:
            self.shield = 4
            print('{:.0f} {:.0f} SHIELD{}'.format(px, py, pod_msgs[self.id]))
        else:
            print('{:.0f} {:.0f} {:.0f}{}'.format(px, py, power, pod_msgs[self.id]))
        self.timeout -= 1


class GameState:
    def __init__(self, laps, checkpoints):
        self.laps = laps
        self.checkpoints = checkpoints
        self.pods = [Pod(i) for i in range(4)]
        self.first_turn = True

    @classmethod
    def read_initial(cls):
        laps = int(input())
        checkpoint_count = int(input())
        checkpoints = [Point(*[int(j) for j in input().split()]) for i in range(checkpoint_count)]
        return cls(laps, checkpoints)

    def read_turn(self):
        for pod in [self.pods[1], self.pods[0], self.pods[3], self.pods[2]]:
            pod.read_turn()

        if self.first_turn:
            angle = math.atan2(self.checkpoints[1].y - self.checkpoints[0].y,
                               self.checkpoints[1].x - self.checkpoints[0].x)
            angle = angle * 180.0 / math.pi
            for pod in self.pods:
                pod.angle = angle
            self.first_turn = False

    def extract_state(self):
        features = [self.laps, len(self.checkpoints)]
        for pod in self.pods:
            features += [
                pod.x / 5000.0,
                pod.y / 5000.0,
                pod.vx / 5000.0,
                pod.vy / 5000.0,
                pod.angle / 360.0,
                float(pod.boost_available),
                pod.timeout / 100.0,
                pod.shield / 4.0,
                float(pod.lap),
                float(pod.next_check_point_id),
            ]
            for i in range(3):
                cp = pod.next_checkpoint(self, i)
                features += [
                    cp.x / 5000.0,
                    cp.y / 5000.0,
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
    IS_CODINGAME = True
    main()
