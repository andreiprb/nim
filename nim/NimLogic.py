class NimLogic(object):

    @staticmethod
    def available_actions(state):
        """
        :param state: pile counts
        :return: available actions (pile, count)
        """
        actions = set()
        for i, pile in enumerate(state):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @staticmethod
    def other_player(player):
        """
        :param player: current player
        :return: other player
        """
        return not player

    @staticmethod
    def nim_sum(state):
        """
        :param state: pile counts
        :return: nim sum (xor on the pile counts)
        """
        nim_sum = 0
        for pile in state:
            nim_sum ^= pile

        return nim_sum

    @staticmethod
    def p_or_n_position(state):
        """
        :param state: pile counts
        :return: 1 for p position, 0 for n position
        """
        if all(pile <= 1 for pile in state):
            one_piles = sum(1 for pile in state if pile == 1)
            return one_piles % 2

        return NimLogic.nim_sum(state) == 0

    @staticmethod
    def heuristic_evaluation(state, player):
        """
        :param state: pile counts
        :param player: current player
        :return: heuristic score
        """
        one_piles = sum(1 for pile in state if pile == 1)
        greater_piles = sum(1 for pile in state if pile > 1)

        if greater_piles == 0:
            if one_piles % 2 == 0:
                score = 50
            else:
                score = -50
        else:
            score = 10 if NimLogic.nim_sum(state) != 0 else -10

        return score if player == 0 else -score
