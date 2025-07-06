import random
import json
import os
from tqdm import tqdm

from Nim.NimLogic import NimLogic
from Nim.NimGameState import NimGameState


class QLearningAgent:
    def __init__(self, misere, pile_count, max_pile, alpha=0.5, epsilon=0.1, gamma=0.9, decay_rate=0.9999, num_episodes=10000):
        self.q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.max_piles = [max_pile] * pile_count
        self.misere = misere
        self.save_path = f"savedAgents/QLearning/qlearning-{pile_count}-{max_pile}-{'misere' if misere else 'normal'}-a{alpha}-e{epsilon}-g{gamma}-d{decay_rate}-ep{num_episodes}.json"

        os.makedirs("savedAgents/QLearning", exist_ok=True)

        if os.path.exists(self.save_path):
            self.load_q_values()
        else:
            self.train()

        print(f"Q-table dimensions: {len(self.q)}")

    def reset_stats(self):
        return

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

    def get_q_value(self, state, action):
        return self.q.get((tuple(state), action), 0)

    def update_q_value(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        max_future_q = self.best_future_reward(next_state)
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q[(tuple(state), action)] = new_q

    def best_future_reward(self, state):
        actions = NimLogic.available_actions(state)
        if not actions:
            return 0

        return max((self.get_q_value(state, action) for action in actions), default=0)

    def choose_action(self, state, is_training=False):
        actions = NimLogic.available_actions(state)
        if not actions:
            return None

        if is_training and random.random() < self.epsilon:
            return random.choice(tuple(actions))

        best_value = float('-inf')
        best_actions = []

        for action in actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def get_move(self, game_state):
        return self.choose_action(game_state.piles)

    def calculate_reward(self, state, next_state, game_over, player_won):
        return 1.0 if game_over and player_won else -1.0

    def train(self):
        for _ in tqdm(range(self.num_episodes)):
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

                self.update_q_value(
                    transition['state'],
                    transition['action'],
                    transition['reward'],
                    transition['next_state']
                )

            self.epsilon *= self.decay_rate

        self.epsilon = self.initial_epsilon
        self.save_q_values()