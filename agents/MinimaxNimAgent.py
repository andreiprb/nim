from nim.NimLogic import NimLogic
import os


class MinimaxNimAgent:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.cache = {}

    def train(self):
        print("Minimax agent initialized with P and N position optimization for misère Nim")

    def choose_action(self, piles, epsilon=None):
        best_score = float('-inf')
        best_action = None

        available_actions = NimLogic.available_actions(piles)

        non_zero_piles = [p for p in piles if p > 0]
        piles_with_one = sum(1 for p in piles if p == 1)
        piles_with_more = sum(1 for p in piles if p > 1)

        if piles_with_more == 0:
            if piles_with_one % 2 == 1:
                for i, pile in enumerate(piles):
                    if pile == 1:
                        return (i, 1)
            else:
                for i, pile in enumerate(piles):
                    if pile == 1:
                        return (i, 1)

        if piles_with_more == 1:
            for i, pile in enumerate(piles):
                if pile > 1:
                    if piles_with_one == 0:
                        return (i, pile - 1)
                    other_nim_sum = 0
                    for j, p in enumerate(piles):
                        if j != i:
                            other_nim_sum ^= p

                    target_size = other_nim_sum

                    if target_size == 0 and pile > 1:
                        return (i, pile - 1)
                    elif target_size < pile:
                        take_count = pile - target_size

                        new_piles = piles.copy()
                        new_piles[i] = target_size

                        new_piles_with_one = sum(1 for p in new_piles if p == 1)
                        new_piles_with_more = sum(1 for p in new_piles if p > 1)

                        if new_piles_with_more == 0 and new_piles_with_one % 2 == 0:
                            continue

                        return (i, take_count)

        nim_sum = self._calculate_nim_sum(piles)

        if nim_sum != 0:
            for pile_idx, pile_size in enumerate(piles):
                if pile_size > 0:
                    target_size = pile_size ^ nim_sum
                    if target_size < pile_size:
                        new_piles = piles.copy()
                        new_piles[pile_idx] = target_size

                        new_piles_with_one = sum(1 for p in new_piles if p == 1)
                        new_piles_with_more = sum(1 for p in new_piles if p > 1)

                        if new_piles_with_more == 0 and new_piles_with_one % 2 == 0:
                            continue

                        return (pile_idx, pile_size - target_size)

        alpha = float('-inf')
        beta = float('inf')

        sorted_actions = sorted(available_actions, key=lambda a: (piles[a[0]], a[1]), reverse=True)

        for action in sorted_actions:
            new_piles = piles.copy()
            pile, count = action
            new_piles[pile] -= count

            score = self._min_value(new_piles, 0, alpha, beta)

            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)

        if best_action:
            return best_action

        return next(iter(available_actions))

    def _calculate_nim_sum(self, piles):
        nim_sum = 0
        for pile in piles:
            nim_sum ^= pile
        return nim_sum

    def _is_terminal(self, piles, depth):
        return all(pile == 0 for pile in piles) or depth >= self.max_depth

    def _get_state_key(self, piles):
        return tuple(piles)

    def _evaluate(self, piles):
        if all(pile == 0 for pile in piles):
            return 1

        nim_sum = self._calculate_nim_sum(piles)
        piles_with_one = sum(1 for p in piles if p == 1)
        piles_with_more = sum(1 for p in piles if p > 1)

        if piles_with_more == 0:
            return 1 if piles_with_one % 2 == 1 else -1

        return 1 if nim_sum != 0 else -1

    def _max_value(self, piles, depth, alpha, beta):
        state_key = self._get_state_key(piles)

        if (state_key, True) in self.cache:
            return self.cache[(state_key, True)]

        if self._is_terminal(piles, depth):
            result = self._evaluate(piles)
            self.cache[(state_key, True)] = result
            return result

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
        state_key = self._get_state_key(piles)

        if (state_key, False) in self.cache:
            return self.cache[(state_key, False)]

        if self._is_terminal(piles, depth):
            result = -self._evaluate(piles)
            self.cache[(state_key, False)] = result
            return result

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