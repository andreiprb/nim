from nim.NimLogic import NimLogic

from agents.Agent import Agent


class NimAgent(Agent):
    def __init__(self):
        super().__init__("Nim")

    def choose_action(self, piles, epsilon=False):
        nim_sum = NimLogic.nim_sum(piles)
        available_actions = NimLogic.available_actions(piles)

        greater_than_one = sum(1 for pile in piles if pile > 1)

        if greater_than_one > 1:
            if nim_sum == 0:
                for i, pile in enumerate(piles):
                    if pile > 0:
                        return i, 1

            for i, pile in enumerate(piles):
                target = pile ^ nim_sum
                to_remove = pile - target
                if to_remove > 0 and (i, to_remove) in available_actions:
                    return i, to_remove

        elif greater_than_one == 1:
            for i, pile in enumerate(piles):
                if pile > 1:
                    return i, pile - 1
        else:
            for i, pile in enumerate(piles):
                if pile == 1:
                    return i, 1
