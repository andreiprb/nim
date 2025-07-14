from math import log2, floor
import numpy as np

from nim.NimLogic import NimLogic


class HelperLogic(object):
    @staticmethod
    def heuristic_evaluation(state, misere):
        one_piles = sum(1 for pile in state if pile == 1)
        greater_piles = sum(1 for pile in state if pile > 1)

        if greater_piles == 0:
            if one_piles % 2 != misere:
                score = -50
            else:
                score = 50
        else:
            score = 10 if NimLogic.nim_sum(state) == 0 else -10

        return score

    @staticmethod
    def canonicalize_state(state):
        indexed_piles = [(pile_size, i) for i, pile_size in enumerate(state)]
        indexed_piles.sort(reverse=True)

        canonical_state = [pile_size for pile_size, _ in indexed_piles]
        index_mapping = [original_idx for _, original_idx in indexed_piles]

        return canonical_state, index_mapping

    @staticmethod
    def map_action_to_original(action, index_mapping):
        canonical_pile_idx, stones = action
        original_pile_idx = index_mapping[canonical_pile_idx]
        return original_pile_idx, stones

    @staticmethod
    def reduce_state(state):
        return state

    @staticmethod
    def generate_sorted_arrays_desc(length, max_val, min_val=0):
        if length == 0:
            yield []
            return

        for first in range(max_val, min_val - 1, -1):
            for rest in HelperLogic.generate_sorted_arrays_desc(length - 1, first, min_val):
                yield [first] + rest