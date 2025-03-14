from nim.NimLogic import NimLogic
import os


class MinimaxNimAgent:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.cache = {}  # For memoization of states

    def train(self):
        # No training needed for minimax, just initialization
        print("Minimax agent initialized with P and N position optimization for misère Nim")

    def choose_action(self, piles, epsilon=None):
        """Choose action using minimax algorithm optimized for misère Nim"""
        best_score = float('-inf')
        best_action = None

        available_actions = NimLogic.available_actions(piles)

        # Count non-zero piles and piles with exactly 1 object
        non_zero_piles = [p for p in piles if p > 0]
        piles_with_one = sum(1 for p in piles if p == 1)
        piles_with_more = sum(1 for p in piles if p > 1)

        # Endgame scenario: when there's only 0s and 1s
        if piles_with_more == 0:
            # If there's an odd number of 1s, take one to leave an even number
            if piles_with_one % 2 == 1:
                for i, pile in enumerate(piles):
                    if pile == 1:
                        return (i, 1)
            # If there's an even number, we're in a losing position, take any
            else:
                for i, pile in enumerate(piles):
                    if pile == 1:
                        return (i, 1)

        # Special case: If there's exactly one pile with more than 1 object
        if piles_with_more == 1:
            # Find that pile
            for i, pile in enumerate(piles):
                if pile > 1:
                    # If there are no other non-zero piles, leave exactly 1
                    if piles_with_one == 0:
                        return (i, pile - 1)
                    # Otherwise, make the nim-sum of the resulting position 0
                    # But we need to consider the special misère rule for the endgame
                    # Calculate the nim-sum of all other piles
                    other_nim_sum = 0
                    for j, p in enumerate(piles):
                        if j != i:
                            other_nim_sum ^= p

                    # Calculate what this pile should be to make nim-sum 0
                    target_size = other_nim_sum

                    # If target_size is 0, we remove the entire pile
                    # If target_size is > pile, we can't make the nim-sum 0
                    if target_size == 0 and pile > 1:
                        # Remove all but 1 to avoid leaving the last object
                        return (i, pile - 1)
                    elif target_size < pile:
                        # This would give us a winning position
                        take_count = pile - target_size

                        # Check if this move would lead to an endgame with all 1s
                        new_piles = piles.copy()
                        new_piles[i] = target_size

                        new_piles_with_one = sum(1 for p in new_piles if p == 1)
                        new_piles_with_more = sum(1 for p in new_piles if p > 1)

                        # If this would lead to an all 1s endgame, make sure it's an odd count
                        if new_piles_with_more == 0 and new_piles_with_one % 2 == 0:
                            # This is a losing position in misère Nim, avoid if possible
                            continue

                        return (i, take_count)

        # Regular play: use nim-sum strategy when not in endgame
        nim_sum = self._calculate_nim_sum(piles)

        if nim_sum != 0:
            for pile_idx, pile_size in enumerate(piles):
                if pile_size > 0:
                    target_size = pile_size ^ nim_sum
                    if target_size < pile_size:
                        # Check if this move would lead to an endgame with all 1s
                        new_piles = piles.copy()
                        new_piles[pile_idx] = target_size

                        new_piles_with_one = sum(1 for p in new_piles if p == 1)
                        new_piles_with_more = sum(1 for p in new_piles if p > 1)

                        # In misère Nim, avoid creating an endgame with even number of 1s
                        if new_piles_with_more == 0 and new_piles_with_one % 2 == 0:
                            # This would be a losing position, check other moves
                            continue

                        return (pile_idx, pile_size - target_size)

        # If no winning move found or we're in a P-position, fall back to minimax
        alpha = float('-inf')
        beta = float('inf')

        # Sort actions to try more promising ones first (for alpha-beta efficiency)
        # Try taking from larger piles first
        sorted_actions = sorted(available_actions, key=lambda a: (piles[a[0]], a[1]), reverse=True)

        for action in sorted_actions:
            # Apply the move
            new_piles = piles.copy()
            pile, count = action
            new_piles[pile] -= count

            # Get score for this move
            score = self._min_value(new_piles, 0, alpha, beta)

            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)

        # If we found a best action, return it
        if best_action:
            return best_action

        # If somehow no best action was found (shouldn't happen), return the first available
        return next(iter(available_actions))

    def _calculate_nim_sum(self, piles):
        """Calculate the nim-sum (XOR of all pile sizes)"""
        nim_sum = 0
        for pile in piles:
            nim_sum ^= pile  # XOR operation
        return nim_sum

    def _is_terminal(self, piles, depth):
        """Check if the state is terminal (all piles are empty or max depth reached)"""
        return all(pile == 0 for pile in piles) or depth >= self.max_depth

    def _get_state_key(self, piles):
        """Generate a hashable key for the piles state"""
        return tuple(piles)

    def _evaluate(self, piles):
        """Evaluate the terminal state for misère Nim"""
        if all(pile == 0 for pile in piles):
            # In misère Nim, the player who would make the next move wins
            # because the previous player took the last object
            return 1

        # For non-terminal states limited by depth, evaluate using nim-sum
        nim_sum = self._calculate_nim_sum(piles)
        piles_with_one = sum(1 for p in piles if p == 1)
        piles_with_more = sum(1 for p in piles if p > 1)

        # In the endgame (only 0s and 1s)
        if piles_with_more == 0:
            return 1 if piles_with_one % 2 == 1 else -1

        # Regular play
        return 1 if nim_sum != 0 else -1

    def _max_value(self, piles, depth, alpha, beta):
        """Maximize the value for the current player"""
        state_key = self._get_state_key(piles)

        # Check cache
        if (state_key, True) in self.cache:
            return self.cache[(state_key, True)]

        # Check if terminal
        if self._is_terminal(piles, depth):
            result = self._evaluate(piles)
            self.cache[(state_key, True)] = result
            return result

        # For non-terminal states, use minimax with alpha-beta pruning
        value = float('-inf')
        available_actions = NimLogic.available_actions(piles)

        for action in available_actions:
            new_piles = piles.copy()
            pile, count = action
            new_piles[pile] -= count

            value = max(value, self._min_value(new_piles, depth + 1, alpha, beta))

            if value >= beta:
                self.cache[(state_key, True)] = value
                return value

            alpha = max(alpha, value)

        self.cache[(state_key, True)] = value
        return value

    def _min_value(self, piles, depth, alpha, beta):
        """Minimize the value for the opponent"""
        state_key = self._get_state_key(piles)

        # Check cache
        if (state_key, False) in self.cache:
            return self.cache[(state_key, False)]

        # Check if terminal
        if self._is_terminal(piles, depth):
            result = -self._evaluate(piles)  # Negate the evaluation
            self.cache[(state_key, False)] = result
            return result

        # For non-terminal states, use minimax with alpha-beta pruning
        value = float('inf')
        available_actions = NimLogic.available_actions(piles)

        for action in available_actions:
            new_piles = piles.copy()
            pile, count = action
            new_piles[pile] -= count

            value = min(value, self._max_value(new_piles, depth + 1, alpha, beta))

            if value <= alpha:
                self.cache[(state_key, False)] = value
                return value

            beta = min(beta, value)

        self.cache[(state_key, False)] = value
        return value