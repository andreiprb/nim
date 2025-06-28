import random
import math
import time
from Nim.NimLogic import NimLogic
from Agents.Agent import Agent


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state.copy()
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = list(NimLogic.available_actions(state))
        random.shuffle(self.untried_actions)

    def uct_select_child(self, exploration_weight=2):
        log_visits = math.log(self.visits) if self.visits > 0 else 0

        best_score = float('-inf')
        best_child = None

        for child in self.children:
            if child.visits == 0:
                return child

            exploitation = child.value / child.visits
            exploration = exploration_weight * math.sqrt(log_visits / child.visits)
            uct_score = exploitation + exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child

    def expand(self):
        if not self.untried_actions:
            return None

        action = self.untried_actions.pop()
        new_state = self.state.copy()
        new_state[action[0]] -= action[1]

        child_node = MCTSNode(new_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.value += result

    def is_terminal(self):
        return all(pile == 0 for pile in self.state)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0


class MCTSAgentV1(Agent):
    def __init__(self, misere, simulation_limit=1000, time_limit=1.0, c_uct=2):
        super().__init__("MCTS")
        self.misere = misere
        self.simulation_limit = simulation_limit
        self.time_limit = time_limit
        self.c_uct = c_uct

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

        root = MCTSNode(state, self.c_uct)

        start_time = time.time()
        simulation_count = 0

        while (time.time() - start_time < self.time_limit and
               simulation_count < self.simulation_limit):

            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.uct_select_child(exploration_weight=c_uct)
                self.nodes_explored += 1

            if not node.is_terminal():
                node = node.expand()
                self.nodes_explored += 1

            result = self._simulate(node.state)

            while node is not None:
                node.update(result)
                node = node.parent
                result = 1 - result

            simulation_count += 1

        best_child = None
        best_visits = -1

        for child in root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_child = child

        if best_child is None and root.children:
            best_child = root.children[0]
        elif not root.children:
            available_actions = list(NimLogic.available_actions(state))
            if available_actions:
                return random.choice(available_actions)
            return None

        return best_child.action

    def _simulate(self, state):
        current_state = state.copy()
        current_player = 0

        while not all(pile == 0 for pile in current_state):
            actions = list(NimLogic.available_actions(current_state))
            if not actions:
                break

            action = random.choice(actions)
            current_state[action[0]] -= action[1]
            current_player = NimLogic.other_player(current_player)

        if self.misere:
            winner = current_player
        else:
            winner = NimLogic.other_player(current_player)

        return 0 if winner == 0 else 1