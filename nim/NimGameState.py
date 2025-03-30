from nim.NimLogic import NimLogic


class NimGameState:
    def __init__(self, initial):
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    def copy(self):
        new_state = NimGameState(self.piles)
        new_state.player = self.player
        new_state.winner = self.winner
        return new_state

    def apply_move(self, action):
        pile, count = action
        new_state = self.copy()

        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        new_state.piles[pile] -= count
        new_state.player = NimLogic.other_player(self.player)

        if all(pile == 0 for pile in new_state.piles):
            new_state.winner = new_state.player

        return new_state