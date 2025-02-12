from nim.NimLogic import NimLogic
from nim.NimGameState import NimGameState

import random


class QLearningAgent:

    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

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

        print("Done training")