from base import BaseAgent


class MCTSAgent(BaseAgent):
    """
    An agent that uses Monte Carlo Tree Search (MCTS) to play nim.
    """
    def __init__(self, misere: bool):
        """
        Initializes the MCTSAgent.
        """
        super().__init__()
        self.misere: bool = misere

    def __str__(self) -> str:
        """
        Returns a string representation of the agent.
        """
        return "MCTS Agent"

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
        Chooses an action based on the current state of the game using MCTS.
        """
        raise NotImplementedError("MCTS logic is not implemented yet.")