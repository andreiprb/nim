from nim.NimLogic import NimLogic


class NimAgent:
    def __init__(self):
        pass

    def train(self):
        # No training needed for this simple agent
        pass

    def choose_action(self, piles, epsilon=False):
        """
        Choose an action using the nim-sum strategy.

        The winning strategy for Nim:
        1. Calculate the nim-sum (XOR of all pile sizes)
        2. If nim-sum is 0, make any move (we're in a losing position)
        3. If nim-sum is not 0, make a move that makes nim-sum 0

        Args:
            piles: List of pile sizes
            epsilon: Not used in this implementation

        Returns:
            (pile_index, count) - The chosen move
        """
        # Calculate nim-sum
        nim_sum = 0
        for pile in piles:
            nim_sum ^= pile

        available_actions = NimLogic.available_actions(piles)

        # If nim-sum is 0, we're in a losing position, make any move
        if nim_sum == 0:
            # Just take 1 from the first non-empty pile
            for i, pile in enumerate(piles):
                if pile > 0:
                    return (i, 1)

        # Otherwise, find a move that makes nim-sum 0
        for i, pile in enumerate(piles):
            if pile > 0:
                # Calculate how many objects to remove to make nim-sum 0
                target = pile ^ (nim_sum ^ pile)
                to_remove = pile - target

                # If this is a valid move, take it
                if to_remove > 0 and (i, to_remove) in available_actions:
                    return (i, to_remove)

        # Fallback: take 1 from the first non-empty pile
        # (shouldn't reach here if the agent is in a winning position)
        for i, pile in enumerate(piles):
            if pile > 0:
                return (i, 1)