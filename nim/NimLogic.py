import numpy as np


class NimLogic:
    """
    Contains the logic for the Nim game, including state manipulation and action selection.
    """
    @staticmethod
    def available_actions(state: list[int]) -> set[tuple[int, int]]:
        """
        Returns a set of available actions for the given Nim game state.
        Each action is represented as a tuple (pile_index, count), where pile_index is the index of the pile.
        """
        actions: set[tuple[int, int]] = set()

        for i, pile in enumerate(state):
            for j in range(1, pile + 1):
                actions.add((i, j))

        return actions

    @staticmethod
    def random_action(actions: set[tuple[int, int]]) -> tuple[int, int]:
        """
        Returns a random action from the set of available actions.
        """
        actions_list: list[tuple[int, int]] = list(actions)
        idx: int = np.random.randint(len(actions_list))
        return actions_list[idx]

    @staticmethod
    def random_action_from_state(state: list[int]) -> tuple[int, int]:
        """
        Returns a random action based on the current state of the game.
        """
        actions: set[tuple[int, int]] = NimLogic.available_actions(state)
        return NimLogic.random_action(actions)

    @staticmethod
    def other_player(player: int) -> int:
        """
        Returns the other player in the game.
        """
        return 1 - player

    @staticmethod
    def nim_sum(state: list[int]) -> int:
        """
        Computes the Nim-sum of the given state.
        """
        return np.bitwise_xor.reduce(state)

    @staticmethod
    def is_p_position(state: list[int], misere: bool) -> bool:
        """
        Determines if the given state is a P-position in Nim.
        """
        if misere and not any(pile > 1 for pile in state):
            return bool(sum(1 for pile in state if pile == 1) % 2)

        return NimLogic.nim_sum(state) == 0