import random
import json
import os
import numpy as np
from agents.Agent import Agent
from nim.NimLogic import NimLogic
from nim.NimGameState import NimGameState


class QLearningAgentV1(Agent):
    def __init__(self, misere, initial_piles, alpha=0.5, epsilon=0.1, gamma=0.9, decay_rate=0.9999):
        super().__init__("QLearningAgent")
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.initial_piles = initial_piles
        self.misere = misere
        self.save_path = f"savedAgents/qlearning-{'-'.join(str(p) for p in initial_piles)}-{misere}.json"

        os.makedirs("savedAgents", exist_ok=True)

        if os.path.exists(self.save_path):
            self.load_q_values()
        else:
            self.train()

    def save_q_values(self):
        serializable_q = {}
        for (state, action), value in self.q.items():
            state_str = ','.join(map(str, state))
            action_str = f"{action[0]},{action[1]}"
            key = f"{state_str}|{action_str}"
            serializable_q[key] = value

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
            self.q = dict()

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

        max_q = float('-inf')
        for action in actions:
            q_value = self.get_q_value(state, action)
            max_q = max(max_q, q_value)

        return max_q if max_q != float('-inf') else 0

    def choose_action(self, state, is_training=False):
        actions = NimLogic.available_actions(state)

        if not actions:
            return None

        if is_training and random.random() < self.epsilon:
            return random.choice(tuple(actions))

        max_q = float('-inf')
        best_actions = []

        for action in actions:
            q_value = self.get_q_value(state, action)

            if q_value > max_q:
                max_q = q_value
                best_actions = [action]
            elif q_value == max_q:
                best_actions.append(action)

        return random.choice(best_actions)

    def get_move(self, game_state):
        return self.choose_action(game_state.piles)

    def calculate_reward(self, state, next_state, game_over, player_won):
        if not game_over:
            return -0.01
        else:
            return 1.0 if player_won else -1.0

    def train(self, num_episodes=50000):
        print(f"Training Q-Learning agent for {num_episodes} episodes...")

        for i in range(num_episodes):
            state = NimGameState(self.initial_piles.copy(), self.misere)

            episode_history = []

            while state.winner is None:
                current_piles = state.piles.copy()
                current_player = state.player

                action = self.choose_action(current_piles, is_training=True)

                next_state = state.apply_move(action)

                game_over = next_state.winner is not None

                player_won = False
                if game_over:
                    if self.misere:
                        player_won = (next_state.winner != current_player)
                    else:
                        player_won = (next_state.winner == current_player)

                reward = self.calculate_reward(
                    current_piles,
                    next_state.piles,
                    game_over,
                    player_won
                )

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

            if (i + 1) % 1000 == 0:
                print(f"Episode {i + 1}/{num_episodes} - Epsilon: {self.epsilon:.6f}")

        self.epsilon = self.initial_epsilon

        self.save_q_values()

        print(f"Training completed.")