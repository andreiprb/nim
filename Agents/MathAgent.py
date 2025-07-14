import numpy as np

from Nim.NimLogic import NimLogic


class MathAgent:
    def __init__(self, misere):
        self.misere = misere

    def __str__(self):
        return "Math Agent"

    def reset_stats(self):
        pass

    def choose_action(self, state):
        xor = np.bitwise_xor.reduce(state)

        if self.misere:
            big = [i for i, h in enumerate(state) if h > 1]

            if len(big) == 1:
                i = big[0]
                count = int(np.count_nonzero(np.array(state) > 0))

                return i, state[i] - int(count % 2)


        for i, h in enumerate(state):
            if h ^ xor < h:
                return i, int(h - (h ^ xor))

        return NimLogic.random_action_from_state(state)


