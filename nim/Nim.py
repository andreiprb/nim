from nim.NimGameState import NimGameState
from nim.NimLogic import NimLogic


class Nim:
    def __init__(self, initial):
        self.state = NimGameState(initial)

    @property
    def piles(self):
        return self.state.piles

    @property
    def player(self):
        return self.state.player

    @property
    def winner(self):
        return self.state.winner

    def move(self, action):
        self.state = self.state.apply_move(action)