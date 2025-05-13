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
    def heuristic_evaluation(state, player, misere):
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