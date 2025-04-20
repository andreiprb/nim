from nim.NimGameState import NimGameState


class Nim:
    def __init__(self, initial, misere=True):
        self.state = NimGameState(initial, misere)
        self.misere = misere

        print("Playing misere Nim!" if self.misere else "Playing normal Nim!")

    @property
    def piles(self):
        return self.state.piles

    @property
    def player(self):
        return self.state.player

    @property
    def winner(self):
        return self.state.winner

    @property
    def is_misere(self):
        return self.misere

    def move(self, action):
        self.state = self.state.apply_move(action)