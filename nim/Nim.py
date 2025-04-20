from nim.NimGameState import NimGameState
from nim.NimLogic import NimLogic


class Nim:
    def __init__(self, initial, misere):
        self.state = NimGameState(initial)
        self.misere = misere

    @property
    def piles(self):
        return self.state.piles

    @property
    def player(self):
        return self.state.player

    @property
    def winner(self):
        if not self.misere:
            return NimLogic.other_player(self.state.winner)

        return self.state.winner

    def move(self, action):
        self.state = self.state.apply_move(action)