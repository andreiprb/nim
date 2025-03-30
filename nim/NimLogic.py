class NimLogic(object):

    @staticmethod
    def available_actions(piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @staticmethod
    def other_player(player):
        return not player

    @staticmethod
    def p_or_n_position(state):
        nim_sum = 0
        for pile in state:
            nim_sum ^= pile

        if all(pile <= 1 for pile in state):
            one_piles = sum(1 for pile in state if pile == 1)
            return one_piles % 2

        return nim_sum == 0
