from typing import Generator

from nim import NimLogic


class HelperLogic:
    @staticmethod
    def heuristic_evaluation(state: list[int], misere: bool) -> int:
        """
        Evaluates the given Nim game state and returns a heuristic score.
        Used for the minimax algorithm to determine the desirability of the state.
        """
        greater_piles: int = sum(1 for pile in state if pile > 1)
        score: int

        if greater_piles == 0:
            one_piles: int = sum(1 for pile in state if pile == 1)

            if one_piles % 2 != misere:
                score = -50
            else:
                score = 50
        else:
            score = 10 if NimLogic.nim_sum(state) == 0 else -10

        return score

    @staticmethod
    def canonicalize_state(state: list[int]) -> tuple[list[int], list[int]]:
        """
        Converts the given state into a canonical form for consistent representation.
        Used for the q-learning algorithm to ensure that equivalent states are treated the same.
        """
        indexed_piles: list[tuple[int, int]] = [(pile_size, i) for i, pile_size in enumerate(state)]
        indexed_piles.sort(reverse=True)

        canonical_state: list[int] = [pile_size for pile_size, _ in indexed_piles]
        index_mapping: list[int] = [original_idx for _, original_idx in indexed_piles]

        return canonical_state, index_mapping

    @staticmethod
    def map_action_to_original(action: tuple[int, int], index_mapping: list[int]) -> tuple[int, int]:
        """
        Maps a canonical action back to the original pile index.
        Used for the q-learning algorithm when actions are represented in a canonical form.
        """
        canonical_pile_idx: int
        stones: int
        canonical_pile_idx, stones = action

        original_pile_idx: int = index_mapping[canonical_pile_idx]

        return original_pile_idx, stones

    @staticmethod
    def reduce_state(state: list[int]) -> list[int]:
        """
        Reduces the state to a simpler form if necessary.
        Used for the q-learning algorithm to minimize the state space.
        """

        # TODO
        return state

    @staticmethod
    def generate_sorted_arrays_desc(length: int, max_val: int, min_val: int = 0) -> Generator[list[int], None, None]:
        """
        Generates all sorted arrays of a given length in descending order.
        Used for the q-learning algorithm to enhance the training process by providing a variety of states.
        """
        if length == 0:
            yield []
            return

        for first in range(max_val, min_val - 1, -1):
            for rest in HelperLogic.generate_sorted_arrays_desc(length - 1, first, min_val):
                yield [first] + rest