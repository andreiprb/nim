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
        nim_sum = 0
        for pile in state:
            nim_sum ^= pile

        return nim_sum

    @staticmethod
    def p_or_n_position(state):
        if all(pile <= 1 for pile in state):
            one_piles = sum(1 for pile in state if pile == 1)
            return one_piles % 2

        return NimLogic.nim_sum(state) == 0

    @staticmethod
    def heuristic_evaluation(state, player, misere):
        one_piles = sum(1 for pile in state if pile == 1)
        greater_piles = sum(1 for pile in state if pile > 1)

        if greater_piles == 0:
            if (one_piles % 2 == 0 and state) or (one_piles % 2 == 1 and not state):
                score = 50
            else:
                score = -50
        else:
            score = 10 if NimLogic.nim_sum(state) != 0 else -10

        return score if player == 0 else -score
