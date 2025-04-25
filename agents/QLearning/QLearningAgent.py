import random
import json
import os
import numpy as np
from agents.Agent import Agent
from nim.NimLogic import NimLogic
from nim.NimGameState import NimGameState


class QLearningAgent(Agent):
    def __init__(self, misere, initial, alpha=0.5, epsilon=0.1, gamma=0.9, decay_rate=0.9999):
        """
        Initialize the Q-Learning Agent.

        Args:
            misere: Whether we're playing misere Nim (lose if you take the last token)
            initial: Initial pile configuration
            alpha: Learning rate (default 0.5)
            epsilon: Exploration rate (default 0.1)
            gamma: Discount factor for future rewards (default 0.9)
            decay_rate: Rate at which epsilon decays during training (default 0.9999)
        """
        super().__init__("QLearningAgent")
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.initial_piles = initial
        self.misere = misere
        self.save_path = f"savedAgents/qlearning-{'-'.join(str(p) for p in initial)}-{misere}.json"

        os.makedirs("savedAgents", exist_ok=True)

        if os.path.exists(self.save_path):
            self.load_q_values()
        else:
            self.train()

    def save_q_values(self):
        """Save Q-values to a JSON file for later use."""
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
        """Load Q-values from a JSON file."""
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
        """
        Get the Q-value for a state-action pair.

        Args:
            state: The pile configuration
            action: The action (pile, count)

        Returns:
            The Q-value for the state-action pair
        """
        return self.q.get((tuple(state), action), 0)

    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value for a state-action pair using the Q-learning update rule.

        Args:
            state: The current state
            action: The action taken
            reward: The immediate reward
            next_state: The resulting state
        """
        old_q = self.get_q_value(state, action)

        # Get the maximum Q-value for the next state
        max_future_q = self.best_future_reward(next_state)

        # Q-learning update rule with discount factor
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)

        self.q[(tuple(state), action)] = new_q

    def best_future_reward(self, state):
        """
        Get the maximum Q-value for a state across all possible actions.

        Args:
            state: The pile configuration

        Returns:
            The maximum Q-value for the state
        """
        # Get all valid actions for the state
        actions = NimLogic.available_actions(state)

        if not actions:  # If no actions are available, return 0
            return 0

        # Find the maximum Q-value among valid actions
        max_q = float('-inf')
        for action in actions:
            q_value = self.get_q_value(state, action)
            max_q = max(max_q, q_value)

        # If no Q-values have been learned yet, return 0
        return max_q if max_q != float('-inf') else 0

    def choose_action(self, state, is_training=False):
        """
        Choose an action using an epsilon-greedy policy.

        Args:
            state: The pile configuration
            is_training: Whether we're in training mode

        Returns:
            The chosen action
        """
        actions = NimLogic.available_actions(state)

        # If no actions are available, return None
        if not actions:
            return None

        # Explore: choose a random action
        if is_training and random.random() < self.epsilon:
            return random.choice(tuple(actions))

        # Exploit: choose the action with the highest Q-value
        max_q = float('-inf')
        best_actions = []

        for action in actions:
            q_value = self.get_q_value(state, action)

            if q_value > max_q:
                max_q = q_value
                best_actions = [action]
            elif q_value == max_q:
                best_actions.append(action)

        # If all actions have the same Q-value (or no Q-values learned yet),
        # choose randomly among them
        return random.choice(best_actions)

    def get_move(self, game_state):
        """
        Choose a move for the current state of the game.

        Args:
            game_state: The current game state

        Returns:
            The chosen action
        """
        return self.choose_action(game_state.piles)

    def calculate_reward(self, state, next_state, game_over, player_won):
        """
        Calculate the reward for transitioning from one state to another.

        Args:
            state: The current state
            next_state: The next state
            game_over: Whether the game is over
            player_won: Whether the player won the game

        Returns:
            The reward for the transition
        """
        if not game_over:
            # Small negative reward for each move to encourage quicker wins
            return -0.01
        else:
            # Game is over, reward based on whether the player won
            return 1.0 if player_won else -1.0

    def train(self, num_episodes=50000):
        """
        Train the Q-learning agent by playing against itself.

        Args:
            num_episodes: The number of episodes to train for
        """
        print(f"Training Q-Learning agent for {num_episodes} episodes...")

        for i in range(num_episodes):
            # Reset the game state
            state = NimGameState(self.initial_piles.copy(), self.misere)

            # Episode history to store state transitions for batch updates
            episode_history = []

            # Play the game until it's over
            while state.winner is None:
                current_piles = state.piles.copy()
                current_player = state.player

                # Choose an action using the epsilon-greedy policy
                action = self.choose_action(current_piles, is_training=True)

                # Apply the action to get the next state
                next_state = state.apply_move(action)

                # Check if the game is over
                game_over = next_state.winner is not None

                # Determine if the current player won
                player_won = False
                if game_over:
                    if self.misere:
                        # In misere Nim, the player who takes the last token loses
                        player_won = (next_state.winner != current_player)
                    else:
                        # In standard Nim, the player who takes the last token wins
                        player_won = (next_state.winner == current_player)

                # Calculate the reward
                reward = self.calculate_reward(
                    current_piles,
                    next_state.piles,
                    game_over,
                    player_won
                )

                # Store the transition
                episode_history.append({
                    'state': current_piles,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.piles,
                    'game_over': game_over
                })

                # Update the game state
                state = next_state

            # Process the episode history to propagate rewards backward
            # Process in reverse order to propagate rewards backward
            for j in range(len(episode_history) - 1, -1, -1):
                transition = episode_history[j]

                # If not the last transition, add the discounted next reward
                if j < len(episode_history) - 1:
                    transition['reward'] += self.gamma * episode_history[j + 1]['reward']

                # Update the Q-value
                self.update_q_value(
                    transition['state'],
                    transition['action'],
                    transition['reward'],
                    transition['next_state']
                )

            # Decay epsilon to reduce exploration over time
            self.epsilon *= self.decay_rate

            # Print progress
            if (i + 1) % 1000 == 0:
                print(f"Episode {i + 1}/{num_episodes} - Epsilon: {self.epsilon:.6f}")

        # Reset epsilon for future use
        self.epsilon = self.initial_epsilon

        # Save the learned Q-values
        self.save_q_values()

        print(f"Training completed.")