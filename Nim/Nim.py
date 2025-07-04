from Nim.NimGameState import NimGameState



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
        player1.reset_stats()
        player2.reset_stats()

        players = [player1, player2]

        if verbose:
            print(f"{'Misere' if self.misere else 'Normal'} game")

        while self.winner is None:
            if verbose:
                print(f"Piles: {self.piles}")

            current_player = self.player
            current_agent = players[current_player]

            pile, count = current_agent.choose_action(self.piles)

            if verbose:
                print(f"Player {int(current_player) + 1} ({current_agent.__class__.__name__}) takes {count} from pile {pile}")

            self.move((pile, count))

            if self.winner is not None:
                if verbose:
                    print(f"Player {int(self.winner) + 1} ({players[self.winner].__class__.__name__}) wins!\n")

        return self.winner
