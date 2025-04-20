from nim.NimLogic import NimLogic

from agents.Agent import Agent


class MinimaxAgent(Agent):
    def __init__(self, max_depth=100):
        super().__init__("Minimax")
        self.max_depth = min(max_depth, 1)
        self.default = 100

    def choose_action(self, state):
        _, best_action = self._minimax(state, 0, float('-inf'), float('inf'), 0)
        return best_action

    def _minimax(self, state, player, alpha, beta, depth):
        if all(pile == 0 for pile in state):
            return (self.default - depth if player == 0 else depth - self.default), None

        if self.max_depth is not None and depth >= self.max_depth:
            heuristic_score = NimLogic.heuristic_evaluation(state, player)
            return heuristic_score, None

        actions = NimLogic.available_actions(state)
        best_action = None

        if player == 0:
            value = float('-inf')
            for action in actions:
                new_state = state.copy()
                new_state[action[0]] -= action[1]

                new_value, _ = self._minimax(new_state, 1, alpha, beta, depth + 1)
                if new_value > value:
                    value = new_value
                    best_action = action

                alpha = max(alpha, value)
                if beta <= alpha:
                    break

            return value, best_action
        else:
            value = float('inf')
            for action in actions:
                new_state = state.copy()
                new_state[action[0]] -= action[1]

                new_value, _ = self._minimax(new_state, 0, alpha, beta, depth + 1)
                if new_value < value:
                    value = new_value
                    best_action = action

                beta = min(beta, value)
                if beta <= alpha:
                    break

            return value, best_action
