from math import log2, floor

from Nim.NimLogic import NimLogic


class HelperLogic(object):
    @staticmethod
    def canonicalize_state(state):
        indexed_piles = [(pile_size, i) for i, pile_size in enumerate(state)]
        indexed_piles.sort(reverse=True)

        canonical_state = [pile_size for pile_size, _ in indexed_piles]
        index_mapping = [original_idx for _, original_idx in indexed_piles]

        return canonical_state, index_mapping

    @staticmethod
    def bubble_up(S, index, mapping):
        while index < len(S) - 1 and S[index] <= S[index + 1]:
            S[index], S[index + 1] = S[index + 1], S[index]
            mapping[index], mapping[index + 1] = \
                mapping[index + 1], mapping[index]
            index += 1
        return index

    @staticmethod
    def reduce_state(state, mapping):
        if not state or all(pile <= 1 for pile in state):
            return state, mapping

        max_power = floor(log2(max(state)))

        changed = True
        while changed:
            changed = False

            for p in range(max_power, -1, -1):
                power_of_two = 1 << p

                while True:
                    indices = [i for i in range(len(state)) if state[i] >= power_of_two and state[i] & power_of_two]

                    if len(indices) < 2:
                        break

                    i, j = indices[0], indices[-1]

                    if i == j:
                        break

                    if sum(state) == 2 * power_of_two:
                        return state, mapping

                    state[i] -= power_of_two
                    state[j] -= power_of_two

                    HelperLogic.bubble_up(state, j, mapping)
                    HelperLogic.bubble_up(state, i, mapping)

                    changed = True

        return state, mapping

    @staticmethod
    def map_action_to_original(action, index_mapping):
        canonical_pile_idx, stones = action
        original_pile_idx = index_mapping[canonical_pile_idx]
        return original_pile_idx, stones

    @staticmethod
    def generate_sorted_arrays(length, max_val, min_val=0):
        if length == 0:
            yield []
            return

        for first in range(min_val, max_val + 1):
            for rest in HelperLogic.generate_sorted_arrays(length - 1, max_val, first):
                yield [first] + rest

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