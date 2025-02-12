from nim.NimLogic import NimLogic
from nim.NimGameState import NimGameState

import random, json, os


class QLearningAgent:
    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon
        self.save_path = "savedAgents/qlearning.json"

        os.makedirs("savedAgents", exist_ok=True)

        if os.path.exists(self.save_path):
            self.load_q_values()

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
                state_str, action_str = key.split('|')
                state = tuple(map(int, state_str.split(',')))
                action = tuple(map(int, action_str.split(',')))
                self.q[(state, action)] = value

            print(f"Q-values loaded from {self.save_path}")
        except Exception as e:
            print(f"Error loading Q-values: {e}")
            self.q = dict()

    def update(self, old_state, action, new_state, reward):
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        return self.q.get((tuple(state), action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        self.q[(tuple(state), action)] = old_q + self.alpha * (reward + future_rewards - old_q)

    def best_future_reward(self, state):
        best = 0

        for key, value in self.q.items():
            if key[0] == tuple(state) and value > best:
                best = value

        return best

    def choose_action(self, state, epsilon=True):
        best = (0, 0)

        actions = NimLogic.available_actions(state)

        for action in actions:
            q = self.q.get((tuple(state), action), 0)

            if q > best[0]:
                best = (q, action)

        if best[0] == 0:
            return random.choice(tuple(actions))

        if epsilon and random.random() < self.epsilon:
            return random.choice(tuple(actions))

        return best[1]

    def train(self, num_episodes=10000):
        if os.path.exists(self.save_path):
            print(f"Q-values already exist at {self.save_path}. Skipping training.")
            return

        print(f"Training Q-Learning agent for {num_episodes} episodes...")

        for i in range(num_episodes):
            state = NimGameState()

            last = {
                0: {"state": None, "action": None},
                1: {"state": None, "action": None}
            }

            while True:
                current_piles = state.piles.copy()
                action = self.choose_action(current_piles)

                last[state.player]["state"] = current_piles
                last[state.player]["action"] = action

                state = state.apply_move(action)

                if state.winner is not None:
                    self.update(current_piles, action, state.piles, -1)
                    self.update(
                        last[state.player]["state"],
                        last[state.player]["action"],
                        state.piles,
                        1
                    )
                    break

                elif last[state.player]["state"] is not None:
                    self.update(
                        last[state.player]["state"],
                        last[state.player]["action"],
                        state.piles,
                        0
                    )

            if (i + 1) % 1000 == 0:
                print(f"Completed training episode {i + 1}")

        self.save_q_values()
