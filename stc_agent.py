import base64
import itertools
import sys
import time

import numpy as np


MODEL_DATA = {}


IS_CODINGAME = False
msg = ''


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
        obz = np.asarray(state)
        state_mean = self.weights['pi/obfilter/runningsum:0'] / self.weights['pi/obfilter/count:0']
        state_std = np.sqrt(np.maximum(
            (self.weights['pi/obfilter/runningsumsq:0'] / self.weights['pi/obfilter/count:0']) - np.square(state_mean),
            1e-2
        ))
        obz = np.clip((obz - state_mean) / state_std, -5.0, 5.0)
        obz = np.asarray(obz, dtype=np.float32)

        action = np.tanh(_matmul(obz, self.weights['pi/pol/fc1/kernel:0']) + self.weights['pi/pol/fc1/bias:0'])
        action = np.tanh(_matmul(action, self.weights['pi/pol/fc2/kernel:0']) + self.weights['pi/pol/fc2/bias:0'])
        action = _matmul(action, self.weights['pi/pol/final/kernel:0']) + self.weights['pi/pol/final/bias:0']

        if IS_CODINGAME:
            if any('/vf/' in tensor_code for tensor_code in self.weights):
                vf = np.tanh(_matmul(obz, self.weights['pi/vf/fc1/kernel:0']) + self.weights['pi/vf/fc1/bias:0'])
                vf = np.tanh(_matmul(vf, self.weights['pi/vf/fc2/kernel:0']) + self.weights['pi/vf/fc2/bias:0'])
                vf = float(_matmul(vf, self.weights['pi/vf/final/kernel:0']) + self.weights['pi/vf/final/bias:0'])
            else:
                vf = 0.0
            global msg
            msg = '{:.2f}'.format(vf)
            print(action, file=sys.stderr)
            print(msg, file=sys.stderr)

        if not self.deterministic:
            action -= np.log(-np.log(np.random.uniform(size=action.shape)))

        return np.argmax(action)

    def compute_action(self, game_state):
        return Action(self.predict(game_state.extract_state()))


class Action:
    outputs = [
        '2 0', '2 1', '2 2', '2 3',
        '3 0', '3 1', '3 2', '3 3',
        '4 0', '4 1', '4 2', '4 3',
        '5 1', '5 2', '5 3',
        '0 0', '0 1', '0 3',
        '1 0', '1 1', '1 2', '1 3',
    ]

    def __init__(self, action):
        self.action = action

    def output(self):
        print(self.outputs[self.action])


colors_mapping = {
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '.': 5,
    '0': 6,
}
read_color = colors_mapping.__getitem__


class Grid:
    def __init__(self):
        self.score = None
        self.grid = None

    def read_turn(self):
        self.score = int(input())
        self.grid = [list(map(read_color, input())) for _ in range(12)]

    def get_features(self):
        return list(itertools.chain(*self.grid))


class GameState:
    def __init__(self):
        self.next_blocks = None
        self.myGrid = Grid()
        self.oppGrid = Grid()

    def read_turn(self):
        self.next_blocks = [tuple(map(read_color, input().split())) for _ in range(8)][::-1]
        self.myGrid.read_turn()
        self.oppGrid.read_turn()

    def extract_state(self):
        features = list(itertools.chain(*self.next_blocks))
        features += self.myGrid.get_features()
        features += self.oppGrid.get_features()
        return features


def main():
    t0 = time.time()
    ai = Model.from_data(MODEL_DATA)
    print('Took {:.2f}ms to load the model'.format((time.time() - t0) * 1000), file=sys.stderr)

    game_state = GameState()
    while True:
        game_state.read_turn()
        t0 = time.time()
        action = ai.compute_action(game_state)
        print('Took {:.2f}ms to predict the action'.format((time.time() - t0) * 1000), file=sys.stderr)
        action.output()


if __name__ == "__main__":
    IS_CODINGAME = True
    main()
