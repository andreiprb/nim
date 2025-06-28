from Nim.NimLogic import NimLogic

from Agents.Agent import Agent

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward network with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class AlphaZeroAttentionNet(nn.Module):
    def __init__(self, num_piles=4, max_pile_size=7, history_length=1, embed_dim=64, num_heads=4,
                 num_attention_layers=2):
        super(AlphaZeroAttentionNet, self).__init__()
        self.num_piles = num_piles
        self.max_pile_size = max_pile_size
        self.history_length = history_length
        self.embed_dim = embed_dim

        # Input size includes current state + previous states
        self.state_size = num_piles * (max_pile_size + 1)
        self.input_size = self.state_size * (history_length + 1)  # +1 for current state

        # Initial embedding layer - maps the flattened input to a sequence of tokens
        self.embedding = nn.Linear(self.input_size, num_piles * embed_dim)

        # Position encoding to help the model understand the order of piles
        self.position_encoding = nn.Parameter(torch.zeros(1, num_piles, embed_dim))
        nn.init.xavier_uniform_(self.position_encoding)

        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads)
            for _ in range(num_attention_layers)
        ])

        # Global pooling
        self.global_pooling = nn.Sequential(
            nn.Linear(num_piles * embed_dim, embed_dim),
            nn.ReLU()
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_piles * max_pile_size)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Initial embedding
        x = self.embedding(x)

        # Reshape to sequence of pile representations
        x = x.view(batch_size, self.num_piles, self.embed_dim)

        # Add position encoding
        x = x + self.position_encoding

        # Prepare for attention (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)

        # Apply attention layers
        for layer in self.attention_layers:
            x = layer(x)

        # Back to (batch_size, seq_len, embed_dim)
        x = x.permute(1, 0, 2)

        # Flatten the sequence dimension
        x = x.reshape(batch_size, -1)

        # Global pooling
        x = self.global_pooling(x)

        # Generate policy and value outputs
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
                    if i < self.num_piles and pile <= self.max_pile_size:  # Ensure we stay within bounds
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
                        if action_idx < policy.size(1):  # Ensure action_idx is valid
                            prior = policy[0, action_idx].item()
                            node.children[action] = MCTSNode(next_state, new_history, prior, node)
                        else:
                            # Use default prior if action_idx is out of bounds
                            node.children[action] = MCTSNode(next_state, new_history, 0.1, node)
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


class AlphaZeroAttentionAgent(Agent):
    def __init__(self, num_piles=7, max_pile_size=11, history_length=3, embed_dim=32, num_heads=2,
                 num_attention_layers=2):
        super().__init__("AlphaZeroAttention")
        self.model = AlphaZeroAttentionNet(num_piles, max_pile_size, history_length, embed_dim, num_heads,
                                           num_attention_layers)
        self.mcts = MCTS(self.model, num_simulations=50)  # Reduced simulations for faster training
        self.save_path = "savedAgents/alphazero_attention.pth"
        self.game_history = []

        os.makedirs("savedAgents", exist_ok=True)

        if os.path.exists(self.save_path):
            self.load_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.save_path))
            self.model.eval()
            print(f"Model loaded from {self.save_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model instead...")

    def choose_action(self, state, epsilon=None):
        # Make a copy of the state before we use it
        current_state = state.copy()

        # Use the history we've collected
        root = self.mcts.search(current_state, self.game_history)

        visits = {action: child.visit_count for action, child in root.children.items()}
        if not visits:  # If no valid actions found
            valid_actions = list(NimLogic.available_actions(current_state))
            if valid_actions:
                return valid_actions[0]  # Return first valid action
            return (0, 1)  # Fallback action

        chosen_action = max(visits.items(), key=lambda x: x[1])[0]

        # Update game history - add current state before the move
        self.game_history.insert(0, current_state)
        # Keep only history_length states
        self.game_history = self.game_history[:self.model.history_length]

        return chosen_action

    def train(self, num_episodes=200, batch_size=32):
        if os.path.exists(self.save_path):
            print(f"Loading pretrained model from {self.save_path}")
            try:
                self.load_model()
                return
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Training a new model instead...")

        print(f"Training AlphaZero Attention agent for {num_episodes} episodes...")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Use smaller game states for training to speed up the process
        training_states = [[1, 3, 5, 7], [2, 4, 6], [3, 5, 7, 9]]

        for episode in range(num_episodes):
            self.model.train()
            states, histories, policies, values = [], [], [], []

            # Randomly select one of the training states
            initial_state = training_states[episode % len(training_states)]

            # Pad the state to match the expected number of piles
            padded_state = initial_state.copy()
            while len(padded_state) < self.model.num_piles:
                padded_state.append(0)
            padded_state = padded_state[:self.model.num_piles]  # Truncate if too long

            game_states, game_histories, game_policies, game_values = self._self_play(padded_state)
            states.extend(game_states)
            histories.extend(game_histories)
            policies.extend(game_policies)
            values.extend(game_values)

            if len(states) >= batch_size:
                # Take a random sample of states to form a batch
                indices = np.random.choice(len(states), min(batch_size, len(states)), replace=False)

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

                # Make sure policy_out and batch_policies have the same shape
                min_size = min(policy_out.size(1), batch_policies.size(1))
                policy_loss = F.kl_div(
                    torch.log(policy_out[:, :min_size]),
                    batch_policies[:, :min_size],
                    reduction='batchmean'
                )
                value_loss = F.mse_loss(value_out.squeeze(), batch_values)
                total_loss = policy_loss + value_loss

                # Print loss value for debugging
                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}, Loss: {total_loss.item():.4f}")

                total_loss.backward()
                optimizer.step()

            # Save model periodically
            if (episode + 1) % 50 == 0:
                print(f"Completed training episode {episode + 1}")
                self.save_model()

        self.save_model()
        print("Training completed!")

    def _self_play(self, initial_state=None):
        # Use the game's initial state if provided, otherwise use a default state
        if initial_state is None:
            game = [1, 3, 5, 7]  # Default initial game state
        else:
            game = initial_state.copy()  # Use provided initial state

        game_history = []  # Track game history

        states, histories, policies, values = [], [], [], []
        current_player = 0

        max_moves = 50  # Safety limit to prevent infinite loops
        move_count = 0

        while not all(pile == 0 for pile in game) and move_count < max_moves:
            move_count += 1

            # Make a copy of the current state
            current_state = game.copy()

            # Search with current history
            root = self.mcts.search(current_state, game_history)

            # Create policy from MCTS visit counts
            visit_counts = np.zeros(self.model.num_piles * self.model.max_pile_size)
            for action, child in root.children.items():
                action_idx = action[0] * self.model.max_pile_size + (action[1] - 1)
                if action_idx < len(visit_counts):  # Ensure we don't go out of bounds
                    visit_counts[action_idx] = child.visit_count

            # Normalize the policy
            total_visits = visit_counts.sum()
            if total_visits > 0:
                policy = visit_counts / total_visits
            else:
                # If no visits, create a uniform policy over valid actions
                valid_actions = NimLogic.available_actions(current_state)
                for action in valid_actions:
                    action_idx = action[0] * self.model.max_pile_size + (action[1] - 1)
                    if action_idx < len(visit_counts):
                        visit_counts[action_idx] = 1
                total_visits = visit_counts.sum()
                if total_visits > 0:
                    policy = visit_counts / total_visits
                else:
                    policy = np.ones(self.model.num_piles * self.model.max_pile_size) / (
                                self.model.num_piles * self.model.max_pile_size)

            # Store training examples
            states.append(current_state)
            histories.append(game_history.copy())  # Make a copy to preserve the history at this point
            policies.append(torch.tensor(policy, dtype=torch.float32))

            # Choose the most visited action
            if not root.children:
                # If no children (shouldn't happen in a valid game), pick a valid action
                valid_actions = list(NimLogic.available_actions(current_state))
                if not valid_actions:
                    break  # No valid actions, game should be over
                action = valid_actions[0]
            else:
                action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]

            # Update game history before making the move
            game_history.insert(0, current_state)
            game_history = game_history[:self.model.history_length]  # Keep only history_length states

            # Apply the action
            pile, count = action
            if pile < len(game) and game[pile] >= count:
                game[pile] -= count
                current_player = 1 - current_player
            else:
                # Invalid move, shouldn't happen but just in case
                print(f"Warning: Invalid move attempted: ({pile}, {count})")
                break

        # Determine game result and fill in values
        game_result = -1 if current_player == 1 else 1
        values = [game_result * (-1 if i % 2 == 0 else 1) for i in range(len(states))]

        return states, histories, policies, [float(v) for v in values]