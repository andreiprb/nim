from .NimLogic import NimLogic


class NimGameState:
    """
    Represents the state of a nim game.
    """
    def __init__(self, initial: list[int], misere: bool):
        """
        Initializes a new nim game state.
        """
        self.piles: list[int] = initial.copy()
        self.player: int = 0
        self.winner: int | None = None
        self.misere: bool = misere

    def copy(self) -> 'NimGameState':
        """
        Creates a copy of the current game state.
        """
        new_state = NimGameState(self.piles, self.misere)
        new_state.player = self.player
        new_state.winner = self.winner
        return new_state

    def apply_move(self, action: tuple[int, int]) -> 'NimGameState':
        """
        Applies a move to the current game state and returns a new state.
        """
        pile: int
        count: int
        pile, count = action
        self._validate_move(pile, count)

        new_state: 'NimGameState' = self.copy()
        new_state.piles[pile] -= count
        new_state.player = NimLogic.other_player(self.player)

        if all(pile == 0 for pile in new_state.piles):
            if new_state.misere:
                new_state.winner = new_state.player
            else:
                new_state.winner = self.player

        return new_state

    def _validate_move(self, pile: int, count: int) -> None:
        """
        Validates that a move is legal.
        """
        if self.winner is not None:
            raise ValueError("Game already won")
        if pile < 0 or pile >= len(self.piles):
            raise IndexError(f"Invalid pile index: {pile}")
        if count < 1 or count > self.piles[pile]:
            raise ValueError(f"Invalid number of objects: {count}")