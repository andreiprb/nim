from nim.NimLogic import NimLogic


class MinimaxAgent:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def choose_action(self, state, epsilon=None):
        actions = NimLogic.available_actions(state)
        best_value = float('-inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')

        for action in actions:
            new_state = state.copy()
            new_state[action[0]] -= action[1]

            value = self._minimax(new_state, 1, alpha, beta, 1)

            if value > best_value:
                best_value = value
                best_action = action

            alpha = max(alpha, best_value)

            if beta <= alpha:
                break

        return best_action

    def _minimax(self, state, player, alpha, beta, depth):
        if all(pile == 0 for pile in state):
            return 1 if player == 0 else -1

        if self.max_depth is not None and depth >= self.max_depth:
            return sum(state) * (-1 if player == 1 else 1)

        actions = NimLogic.available_actions(state)

        if player == 0:
            value = float('-inf')
            for action in actions:
                new_state = state.copy()
                new_state[action[0]] -= action[1]
                value = max(value, self._minimax(new_state, 1, alpha, beta, depth + 1))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = float('inf')
            for action in actions:
                new_state = state.copy()
                new_state[action[0]] -= action[1]
                value = min(value, self._minimax(new_state, 0, alpha, beta, depth + 1))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def train(self):
        print("Playing versus Minimax agent. Nothing to train.")