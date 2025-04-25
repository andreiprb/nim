import random
import json
import os
from tqdm import tqdm

from agents.Agent import Agent
from nim.NimLogic import NimLogic
from nim.NimGameState import NimGameState


class QLearningAgentV2(Agent):
    def __init__(self, misere, max_piles, alpha=0.5, epsilon=0.1, gamma=0.9, decay_rate=0.9999):
        super().__init__("Q-LearningV2")
        self.q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.max_piles = max_piles
        self.misere = misere
        self.save_path = f"savedAgents/qlearningV2-{'-'.join(str(p) for p in max_piles)}-{misere}.json"

        os.makedirs("savedAgents", exist_ok=True)

        if os.path.exists(self.save_path):
            self.load_q_values()
        else:
            self.train()

        print(f"Q-table dimensions: {len(self.q)}")

    def save_q_values(self):
        serializable_q = {f"{','.join(map(str, state))}|{action[0]},{action[1]}": value
                          for (state, action), value in self.q.items()}

        with open(self.save_path, 'w') as f:
            json.dump(serializable_q, f)
        print(f"Q-values saved to {self.save_path}")

    def load_q_values(self):
        try:
            with open(self.save_path, 'r') as f:
                serialized_q = json.load(f)

            for key, value in serialized_q.items():
                state_part, action_part = key.split('|')
                state = tuple(map(int, state_part.split(',')))
                action = tuple(map(int, action_part.split(',')))
                self.q[(state, action)] = value

            print(f"Q-values loaded from {self.save_path}")
        except Exception as e:
            print(f"Error loading Q-values: {e}")
            self.q = {}

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

    def get_move(self, game_state):
        return self.choose_action(game_state.piles)

    def calculate_reward(self, state, next_state, game_over, player_won):
        return 1.0 if game_over and player_won else -1.0 if game_over else -0.01

    def train(self, num_episodes=50000):
        for _ in tqdm(range(num_episodes)):
            state = NimGameState(self.max_piles.copy(), self.misere)
            episode_history = []

            while state.winner is None:
                current_piles = state.piles.copy()
                current_player = state.player
                action = self.choose_action(current_piles, is_training=True)
                next_state = state.apply_move(action)

                game_over = next_state.winner is not None
                player_won = game_over and ((self.misere and next_state.winner != current_player) or
                                            (not self.misere and next_state.winner == current_player))

                reward = self.calculate_reward(current_piles, next_state.piles, game_over, player_won)

                episode_history.append({
                    'state': current_piles,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.piles,
                    'game_over': game_over
                })

                state = next_state

            for j in range(len(episode_history) - 1, -1, -1):
                transition = episode_history[j]

                if j < len(episode_history) - 1:
                    transition['reward'] += self.gamma * episode_history[j + 1]['reward']

                self.update_q_value(
                    transition['state'],
                    transition['action'],
                    transition['reward'],
                    transition['next_state']
                )

            self.epsilon *= self.decay_rate

        self.epsilon = self.initial_epsilon
        self.save_q_values()