from .NimGameState import NimGameState

from helper import HelperAgent


class NimGame:
    """
    Represents a Nim game with a given initial state and rules.
    """
    def __init__(self, initial_piles: list[int], misere: bool):
        """
        Initializes a new Nim game.
        """
        self.state: NimGameState = NimGameState(initial_piles, misere)
        self.misere: bool = misere

    @property
    def piles(self) -> list[int]:
        """
        Returns the current state of the piles in the game.
        """
        return self.state.piles

    @property
    def player(self) -> int:
        """
        Returns the current player (0 or 1).
        """
        return self.state.player

    @property
    def winner(self) -> int | None:
        """
        Returns the winner of the game, or None if the game is still ongoing.
        """
        return self.state.winner

    @property
    def is_misere(self) -> bool:
        """
        Returns whether the game is played in misère mode.
        """
        return self.misere

    def play(self, player1: HelperAgent, player2: HelperAgent, verbose: bool = False) -> int | None:
        """
        Plays a game of Nim between two players.
        """
        player1.reset_stats()
        player2.reset_stats()

        players: list[HelperAgent] = [player1, player2]

        if verbose:
            print(f"{'Misere' if self.misere else 'Normal'} game")

        while self.winner is None:
            if verbose:
                print(f"Piles: {self.piles}")

            current_player: int = self.player
            current_agent: HelperAgent = players[current_player]

            pile: int
            count: int
            pile, count = current_agent.choose_action(self.piles)

            if verbose:
                self._print_move(current_player, current_agent, pile, count)

            self.state: NimGameState = self.state.apply_move((pile, count))

            if self.winner is not None:
                if verbose:
                    self._print_winner(players)

        return self.winner

    def _print_move(self, current_player: int, current_agent: HelperAgent, pile: int, count: int) -> None:
        """
        Prints the move information if verbose mode is enabled.
        """
        print(f"Player {current_player + 1} ({current_agent.__class__.__name__}) takes {count} from pile {pile}")

    def _print_winner(self, players: list[HelperAgent]) -> None:
        """
        Prints the winner information if verbose mode is enabled.
        """
        print(f"Player {self.winner + 1} ({players[self.winner].__class__.__name__}) wins!\n")
