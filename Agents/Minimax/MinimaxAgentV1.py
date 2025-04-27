from Nim.NimLogic import NimLogic

from Agents.Agent import Agent


class MinimaxAgentV1(Agent):
    def __init__(self, misere, max_depth):
        super().__init__("MinimaxV1")
        self.misere = misere
        self.max_depth = max(max_depth, 1)
        self.default = self.max_depth

        self.nodes_explored = 0
        self.moves_count = 0
        self.mean_nodes = 0

    def reset_stats(self):
        self.nodes_explored = 0
        self.moves_count = 0
        self.mean_nodes = 0

    def compute_mean_nodes(self):
        if self.moves_count == 0:
            return

        self.mean_nodes = self.nodes_explored / self.moves_count

    def choose_action(self, state):
        self.moves_count += 1

        _, best_action = self._minimax(state, 0, float('-inf'), float('inf'), 0)

        return best_action

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
