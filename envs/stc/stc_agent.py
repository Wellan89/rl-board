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
        raise NotImplementedError('Copy me from the policy file')

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
        print(self.outputs[self.action] + (' {}'.format(msg) if msg else ''))


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
