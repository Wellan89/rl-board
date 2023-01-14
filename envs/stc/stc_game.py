import random

from alpha_zero.Game import Game
from cpp import stc_pybind

stc_pybind.srand(random.SystemRandom().getrandbits(32))


class StcGame(Game):
    def getInitBoard(self):
        return stc_pybind.World().compute_state(False)

    def getBoardSize(self):
        raise NotImplementedError

    def getActionSize(self):
        return 22

    def getNextState(self, board, player, action):
        pass

    def getValidMoves(self, board, player):
        pass

    def getGameEnded(self, board, player):
        pass

    def getCanonicalForm(self, board, player):
        pass

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        pass
