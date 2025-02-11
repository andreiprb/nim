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
        return 0 if player == 1 else 1