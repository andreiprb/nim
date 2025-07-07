import numpy as np
from math import log2, floor
from bisect import bisect_right


class NimLogic(object):
    @staticmethod
    def available_actions(state):
        actions = set()
        for i, pile in enumerate(state):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @staticmethod
    def other_player(player):
        return not player

    @staticmethod
    def nim_sum(state):
        return np.bitwise_xor.reduce(state)

    @staticmethod
    def is_p_position(state, misere):
        if misere and not any(pile > 1 for pile in state):
            return sum(1 for pile in state if pile == 1) % 2

        return bool(NimLogic.nim_sum(state)) == 0

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
    def bubble_up(S, index, mapping):
        while index < len(S) - 1 and S[index] <= S[index + 1]:
            S[index], S[index + 1] = S[index + 1], S[index]
            mapping[index], mapping[index + 1] =\
                mapping[index + 1], mapping[index]
            index += 1
        return index

    @staticmethod
    def reduce_state(state, mapping):
        if not state or max(state) == 0:
            return state, mapping

        max_power = floor(log2(max(state)))

        changed = True
        while changed:
            changed = False

            for p in range(max_power, -1, -1):
                power_of_two = 1 << p

                while True:
                    right = bisect_right(state, power_of_two)

                    indices = [i for i in range(right) if state[i] & power_of_two]

                    if len(indices) < 2:
                        break

                    i, j = indices[0], indices[-1]

                    if i == j:
                        break

                    if sum(state) == 2 * power_of_two:
                        return state, mapping

                    state[i] -= power_of_two
                    state[j] -= power_of_two

                    NimLogic.bubble_up(state, j, mapping)
                    NimLogic.bubble_up(state, i, mapping)

                    changed = True

        return state, mapping

    @staticmethod
    def map_action_to_original(action, index_mapping):
        canonical_pile_idx, stones = action
        original_pile_idx = index_mapping[canonical_pile_idx]
        return original_pile_idx, stones