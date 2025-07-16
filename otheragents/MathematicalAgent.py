import numpy as np

from nim.NimLogic import NimLogic

from base.BaseAgent import BaseAgent


class MathematicalAgent(BaseAgent):
    """
    An agent that uses the known mathematical strategy to play Nim.
    """
    def __init__(self, misere):
        """
        Initializes the MathematicalAgent.
        """
        super().__init__()
        self.misere: bool = misere

    def __str__(self) -> str:
        """
        Returns a string representation of the agent.
        """
        return "Algorithmic Agent"

    def reset_stats(self):
        """
        Resets the statistics of the agent.
        """
        pass

    def get_stats(self) -> tuple | None:
        """
        Returns the statistics of the agent.
        """
        return None

    def choose_action(self, state: list[int]) -> tuple[int, int]:
        """
        Chooses an action based on the current state of the game.
        """
        xor: int = np.bitwise_xor.reduce(state)

        if self.misere:
            big: list[int] = [i for i, h in enumerate(state) if h > 1]

            if len(big) == 1:
                i: int = big[0]
                count: int = int(np.count_nonzero(np.array(state) > 0))

                return i, state[i] - int(count % 2)


        for i, h in enumerate(state):
            if h ^ xor < h:
                return i, int(h - (h ^ xor))

        return NimLogic.random_action_from_state(state)


