import random
import os

from agents.QLearning.QLearningAgentV1 import QLearningAgentV1

class QLearningAgentV2(QLearningAgentV1):
    def __init__(self, misere, max_piles, alpha=0.5, epsilon=0.1, gamma=0.9, decay_rate=0.9999):
        super().__init__(misere, max_piles, alpha, epsilon, gamma, decay_rate)
        self.name = "Q-LearningV2"
        self.save_path = f"savedAgents/qlearningV2-{'-'.join(str(p) for p in max_piles)}-{misere}.json"

        self.q = {}

        if os.path.exists(self.save_path):
            self.load_q_values()
        else:
            self.train()

        print(f"Q-table dimensions: {len(self.q)}")

    def normalize_state(self, state):
        indexed_state = list(enumerate(state))
        indexed_state.sort(key=lambda x: x[1])
        original_indices = [i for i, _ in indexed_state]
        sorted_state = tuple(val for _, val in indexed_state)
        return sorted_state, original_indices

    def get_q_value(self, state, action):
        sorted_state, original_indices = self.normalize_state(state)
        original_pile_idx, count = action
        sorted_pile_idx = original_indices.index(original_pile_idx)
        sorted_action = (sorted_pile_idx, count)
        return self.q.get((sorted_state, sorted_action), 0)

    def update_q_value(self, state, action, reward, next_state):
        sorted_state, original_indices = self.normalize_state(state)
        original_pile_idx, count = action
        sorted_pile_idx = original_indices.index(original_pile_idx)
        sorted_action = (sorted_pile_idx, count)

        old_q = self.q.get((sorted_state, sorted_action), 0)
        max_future_q = self.best_future_reward(next_state)
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)

        self.q[(sorted_state, sorted_action)] = new_q

    def best_future_reward(self, state):
        sorted_state, _ = self.normalize_state(state)

        actions = set()
        for sorted_idx, pile in enumerate(sorted_state):
            actions.update((sorted_idx, j) for j in range(1, pile + 1))

        if not actions:
            return 0

        return max((self.q.get((sorted_state, action), 0) for action in actions), default=0)

    def choose_action(self, state, is_training=False):
        sorted_state, index_mapping = self.normalize_state(state)

        sorted_actions = set()
        for sorted_idx, pile in enumerate(sorted_state):
            sorted_actions.update((sorted_idx, j) for j in range(1, pile + 1))

        if not sorted_actions:
            return None

        if is_training and random.random() < self.epsilon:
            sorted_action = random.choice(tuple(sorted_actions))
            original_pile_idx = index_mapping[sorted_action[0]]
            return (original_pile_idx, sorted_action[1])

        best_value = float('-inf')
        best_sorted_actions = []

        for sorted_action in sorted_actions:
            q_value = self.q.get((sorted_state, sorted_action), 0)

            if q_value > best_value:
                best_value = q_value
                best_sorted_actions = [sorted_action]
            elif q_value == best_value:
                best_sorted_actions.append(sorted_action)

        sorted_action = random.choice(best_sorted_actions)
        original_pile_idx = index_mapping[sorted_action[0]]
        return original_pile_idx, sorted_action[1]