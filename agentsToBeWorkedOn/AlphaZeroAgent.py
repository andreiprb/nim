from nim.NimLogic import NimLogic

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os


class AlphaZeroNet(nn.Module):
    def __init__(self, num_piles=4, max_pile_size=7, history_length=1):
        super(AlphaZeroNet, self).__init__()
        self.num_piles = num_piles
        self.max_pile_size = max_pile_size
        self.history_length = history_length

        # Input size now includes current state + previous states
        self.state_size = num_piles * (max_pile_size + 1)
        input_size = self.state_size * (history_length + 1)  # +1 for current state

        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(64, num_piles * max_pile_size)

        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.shared(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return F.softmax(policy, dim=1), value

    def encode_state(self, state, history=None):
        # Encode current state
        encoded = torch.zeros(1, self.state_size * (self.history_length + 1))

        # Fill in current state
        for i, pile in enumerate(state):
            if pile <= self.max_pile_size:
                encoded[0, i * (self.max_pile_size + 1) + pile] = 1

        # Fill in history states if available
        if history is not None:
            for h_idx, h_state in enumerate(history):
                if h_idx >= self.history_length:
                    break
                offset = self.state_size * (h_idx + 1)
                for i, pile in enumerate(h_state):
                    if pile <= self.max_pile_size:
                        encoded[0, offset + i * (self.max_pile_size + 1) + pile] = 1

        return encoded


class MCTSNode:
    def __init__(self, state, history=None, prior=0, parent=None):
        self.state = state
        self.history = history if history is not None else []
        self.prior = prior
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.player = 0 if parent is None else 1 - parent.player

    def value(self):
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def select_child(self, c_puct=1.0):
        best_score = float('-inf')
        best_action = None

        sum_visits = sum(child.visit_count for child in self.children.values())

        for action, child in self.children.items():
            q_value = -child.value()
            u_value = c_puct * child.prior * math.sqrt(sum_visits) / (1 + child.visit_count)
            ucb_score = q_value + u_value

            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action

        return best_action, self.children[best_action]


class MCTS:
    def __init__(self, model, num_simulations=100):
        self.model = model
        self.num_simulations = num_simulations

    def search(self, root_state, history=None):
        root = MCTSNode(root_state, history)

        for _ in range(self.num_simulations):
            node = root
            path = [node]

            while node.children and not self._is_terminal(node.state):
                action, node = node.select_child()
                path.append(node)

            if not self._is_terminal(node.state):
                policy, value = self._evaluate(node.state, node.history)
                valid_actions = NimLogic.available_actions(node.state)

                for action in valid_actions:
                    if action not in node.children:
                        next_state = node.state.copy()
                        next_state[action[0]] -= action[1]

                        # Create new history by adding current state to history
                        new_history = [node.state.copy()]
                        if node.history:
                            new_history.extend(node.history[:self.model.history_length - 1])

                        action_idx = action[0] * self.model.max_pile_size + (action[1] - 1)
                        prior = policy[0, action_idx].item()
                        node.children[action] = MCTSNode(next_state, new_history, prior, node)
            else:
                value = 1.0

            for node in reversed(path):
                node.visit_count += 1
                node.value_sum += value
                value = -value

        return root

    def _evaluate(self, state, history):
        encoded_state = self.model.encode_state(state, history)
        with torch.no_grad():
            policy, value = self.model(encoded_state)
        return policy, value.item()

    def _is_terminal(self, state):
        return all(pile == 0 for pile in state)


class AlphaZeroAgent:
    def __init__(self, num_piles=4, max_pile_size=7, history_length=1):
        self.model = AlphaZeroNet(num_piles, max_pile_size, history_length)
        self.mcts = MCTS(self.model)
        self.save_path = "savedAgents/alphazero_with_history.pth"
        self.game_history = []

        os.makedirs("savedAgents", exist_ok=True)

        if os.path.exists(self.save_path):
            self.load_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        print(f"Model loaded from {self.save_path}")

    def choose_action(self, state, epsilon=None):
        # Make a copy of the state before we use it
        current_state = state.copy()

        # Use the history we've collected
        root = self.mcts.search(current_state, self.game_history)

        visits = {action: child.visit_count for action, child in root.children.items()}
        chosen_action = max(visits.items(), key=lambda x: x[1])[0]

        # Update game history - add current state before the move
        self.game_history.insert(0, current_state)
        # Keep only history_length states
        self.game_history = self.game_history[:self.model.history_length]

        return chosen_action

    def train(self, num_episodes=1000, batch_size=32):
        if os.path.exists(self.save_path):
            return

        print(f"Training AlphaZero agent with history for {num_episodes} episodes...")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for episode in range(num_episodes):
            self.model.train()
            states, histories, policies, values = [], [], [], []

            game_states, game_histories, game_policies, game_values = self._self_play()
            states.extend(game_states)
            histories.extend(game_histories)
            policies.extend(game_policies)
            values.extend(game_values)

            if len(states) >= batch_size:
                indices = np.random.choice(len(states), batch_size)

                # Create batches with history
                batch_encoded_states = [
                    self.model.encode_state(states[i], histories[i])
                    for i in indices
                ]
                batch_encoded_states = torch.cat(batch_encoded_states, dim=0)

                batch_policies = torch.stack([policies[i] for i in indices])
                batch_values = torch.tensor([values[i] for i in indices])

                optimizer.zero_grad()
                policy_out, value_out = self.model(batch_encoded_states)

                policy_loss = F.kl_div(torch.log(policy_out), batch_policies, reduction='batchmean')
                value_loss = F.mse_loss(value_out.squeeze(), batch_values)
                total_loss = policy_loss + value_loss

                total_loss.backward()
                optimizer.step()

            if (episode + 1) % 100 == 0:
                print(f"Completed training episode {episode + 1}")

        self.save_model()

    def _self_play(self):
        game = [1, 3, 5, 7]  # Initial game state
        game_history = []  # Track game history

        states, histories, policies, values = [], [], [], []
        current_player = 0

        while not all(pile == 0 for pile in game):
            # Make a copy of the current state
            current_state = game.copy()

            # Search with current history
            root = self.mcts.search(current_state, game_history)

            # Create policy from MCTS visit counts
            encoded_state = self.model.encode_state(current_state, game_history)
            visit_counts = np.zeros(self.model.num_piles * self.model.max_pile_size)
            for action, child in root.children.items():
                action_idx = action[0] * self.model.max_pile_size + (action[1] - 1)
                visit_counts[action_idx] = child.visit_count

            policy = visit_counts / (visit_counts.sum() or 1)  # Avoid division by zero

            # Store training examples
            states.append(current_state)
            histories.append(game_history.copy())  # Make a copy to preserve the history at this point
            policies.append(torch.tensor(policy))

            # Choose the most visited action
            action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]

            # Update game history before making the move
            game_history.insert(0, current_state)
            game_history = game_history[:self.model.history_length]  # Keep only history_length states

            # Apply the action
            game[action[0]] -= action[1]
            current_player = 1 - current_player

        # Determine game result and fill in values
        game_result = -1 if current_player == 1 else 1
        values = [game_result * (-1 if i % 2 == 0 else 1) for i in range(len(states))]

        return states, histories, policies, values