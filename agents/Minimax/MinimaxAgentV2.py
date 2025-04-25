from nim.NimLogic import NimLogic

from agents.Minimax.MinimaxAgentV1 import MinimaxAgentV1


class MinimaxAgentV2(MinimaxAgentV1):
    def __init__(self, misere, max_depth):
        super().__init__(misere, max_depth)
        self.name = "MinimaxV2"

    def _minimax(self, state, player, alpha, beta, depth):
        self.nodes_explored += 1

        if all(pile == 0 for pile in state):
            sign = 1 if player == self.misere else -1
            return sign * (self.default - depth), None

        if self.max_depth is not None and depth >= self.max_depth:
            heuristic_score = NimLogic.heuristic_evaluation(state, player, self.misere)
            return heuristic_score, None

        actions = NimLogic.available_actions(state)

        best_action = None
        is_endgame = all(pile <= 1 for pile in state)

        if player == 0:
            value = float('-inf')
            for action in actions:
                new_state = state.copy()
                new_state[action[0]] -= action[1]

                is_p_position = NimLogic.p_or_n_position(new_state)

                if is_p_position:
                    if self.misere and is_endgame:
                        return self.default - depth, action
                    else:
                        return depth - self.default, action

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

                is_p_position = NimLogic.p_or_n_position(new_state)

                if is_p_position:
                    if self.misere and is_endgame:
                        return depth - self.default, action
                    else:
                        return self.default - depth, action

                new_value, _ = self._minimax(new_state, 0, alpha, beta, depth + 1)
                if new_value < value:
                    value = new_value
                    best_action = action

                beta = min(beta, value)
                if beta <= alpha:
                    break

            return value, best_action
