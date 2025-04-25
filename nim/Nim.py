from nim.NimGameState import NimGameState


class Nim:
    def __init__(self, initial_piles, misere=True):
        self.state = NimGameState(initial_piles, misere)
        self.misere = misere

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

    def play(self, player1, player2, verbose=True):
        players = [player1, player2]

        if verbose:
            print(f"Initial piles: {self.piles}")
            print(f"{'Misere' if self.misere else 'Normal'} game")

        while self.winner is None:
            current_player = self.player
            current_agent = players[current_player]

            pile, count = current_agent.choose_action(self.piles)

            if verbose:
                print(f"Player {int(current_player) + 1} ({current_agent.name}) takes {count} from pile {pile}")

            self.move((pile, count))

            if verbose:
                print(f"Piles: {self.piles}")

            if self.winner is not None:
                if verbose:
                    print(f"Player {int(self.winner) + 1} ({players[self.winner].name}) wins!")

        return self.winner
