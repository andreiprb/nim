from nim.NimLogic import NimLogic

from base.BaseAgent import BaseAgent


class HumanAgent(BaseAgent):
    """
    An agent that allows a human player to play the game of nim.
    """
    def __init__(self):
        """
        Initializes the HumanAgent.
        """
        super().__init__()

    def __str__(self) -> str:
        """
        Returns a string representation of the agent.
        """
        return "Human Agent"

    def reset_stats(self):
        """
        Resets the statistics of the agent.
        """
        return

    def get_stats(self) -> tuple | None:
        """
        Returns the statistics of the agent.
        """
        return None

    def choose_action(self, piles: list[int]) -> tuple[int, int]:
        """
        Prompts the human player to choose an
        action based on the current piles.
        """
        print("Your Turn")
        available_actions: set[tuple[int, int]] = \
            NimLogic.available_actions(piles)

        while True:
            pile: int = self._get_int("Choose Pile: ")
            count: int = self._get_int("Choose Count: ")

            if (pile, count) in available_actions:
                break

            print("Invalid move, try again.")

        return pile, count

    @staticmethod
    def _get_int(prompt: str) -> int:
        """
        Prompts the user for an integer input
        until a valid integer is provided.
        """
        while True:
            user_input = input(prompt)

            try:
                return int(user_input)
            except ValueError:
                continue

        return -1

