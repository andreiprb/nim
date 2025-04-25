from nim.NimLogic import NimLogic

from agents.Agent import Agent


class MinimaxAgentV2(Agent):
    def __init__(self, misere, max_depth):
        super().__init__("MinimaxV2")
        self.misere = misere
        self.max_depth = max(max_depth, 1)
        self.default = self.max_depth

    def choose_action(self, state):
        _, best_action = self._minimax(state, 0, float('-inf'), float('inf'), 0)
        return best_action

    def _minimax(self, state, player, alpha, beta, depth):
        if all(pile == 0 for pile in state):
            sign = 1 if player == self.misere else -1
            return sign * (self.default - depth), None

        if self.max_depth is not None and depth >= self.max_depth:
            heuristic_score = NimLogic.heuristic_evaluation(state, player)
            return heuristic_score, None

        actions = NimLogic.available_actions(state)
        actions = sorted(actions, key=lambda a: a[1], reverse=True)

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
