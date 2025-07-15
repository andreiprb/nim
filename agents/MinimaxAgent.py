from nim import NimLogic
from helper import HelperLogic

from agents import Agent


class MinimaxAgent(Agent):
    def __init__(self, misere, max_depth, canonical=False, P_pruning=False, aggressive=False):
        super().__init__()

        self.misere = misere
        self.max_depth = max(max_depth, 1)
        self.default = self.max_depth

        self.canonical = canonical
        self.P_pruning = P_pruning
        self.sorting = aggressive

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

        self.mean_nodes = self.nodes_explored / self.moves_count

        return best_action

    def _minimax(self, state, player, alpha, beta, depth):
        self.nodes_explored += 1

        if all(pile == 0 for pile in state):
            sign = -1 if player == self.misere else 1
            return sign * (self.default - depth), None

        if self.max_depth is not None and depth >= self.max_depth:
            heuristic_score = HelperLogic.heuristic_evaluation(state, self.misere)
            return -heuristic_score if player == 0 else heuristic_score, None

        """ CANONICALIZATION OF STATE """
        if self.canonical:
            state, index_mapping = HelperLogic.canonicalize_state(state)

        actions = NimLogic.available_actions(state)

        """ ACTION SORTING FOR AGGRESSIVE PLAYSTYLE """
        if self.sorting:
            actions = sorted(actions, key=lambda x: x[1], reverse=True)

        best_action = None

        if player == 0:
            value = float('-inf')
            for action in actions:
                new_state = state.copy()
                new_state[action[0]] -= action[1]

                """ PRUNING AFTER P_POSITIONS """
                if self.P_pruning and NimLogic.is_p_position(new_state, self.misere):

                    """ MAP ACTION TO ORIGINAL STATE """
                    original_action = action if not self.canonical else HelperLogic.map_action_to_original(action, index_mapping)

                    return (self.default - depth), original_action

                new_value, _ = self._minimax(new_state, 1, alpha, beta, depth + 1)
                if new_value > value:
                    value = new_value

                    """ MAP ACTION TO ORIGINAL STATE """
                    best_action = action if not self.canonical else HelperLogic.map_action_to_original(action, index_mapping)

                alpha = max(alpha, value)
                if beta <= alpha:
                    break

            return value, best_action
        else:
            value = float('inf')
            for action in actions:
                new_state = state.copy()
                new_state[action[0]] -= action[1]

                """ PRUNING AFTER P_POSITIONS """
                if self.P_pruning and NimLogic.is_p_position(new_state, self.misere):

                    """ MAP ACTION TO ORIGINAL STATE """
                    original_action = action if not self.canonical else HelperLogic.map_action_to_original(action, index_mapping)

                    return (depth - self.default), original_action

                new_value, _ = self._minimax(new_state, 0, alpha, beta, depth + 1)
                if new_value < value:
                    value = new_value

                    """ MAP ACTION TO ORIGINAL STATE """
                    best_action = action if not self.canonical else HelperLogic.map_action_to_original(action, index_mapping)

                beta = min(beta, value)
                if beta <= alpha:
                    break

            return value, best_action
