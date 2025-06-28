import numpy as np


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
    def map_action_to_original(action, index_mapping):
        canonical_pile_idx, stones = action
        original_pile_idx = index_mapping[canonical_pile_idx]
        return original_pile_idx, stones