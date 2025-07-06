import random
import json
import os
import numpy as np
from tqdm import tqdm

from Nim.NimLogic import NimLogic
from Nim.NimGameState import NimGameState


class QLearningAgent:
    def __init__(self, misere, pile_count, max_pile, override=False,
                 alpha=0.3, epsilon=0.2, gamma=1.0, num_episodes=50000,
                 canonical=False):
        self.misere = misere
        self.pile_count = pile_count
        self.max_pile = max_pile

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.q = {}

        self.canonical = canonical

        self.save_dir = "../savedAgents/QLearning"
        os.makedirs(self.save_dir, exist_ok=True)
        self.filename = f"qlearning-{pile_count}-{max_pile}-{'misere' if misere else 'normal'}{'-canonical' if canonical else ''}.json"
        self.save_path = os.path.join(self.save_dir, self.filename)

        if override or not self._load():
            self.train(num_episodes)
            self._save()

        print(f"Agent ready. Q-table size: {len(self.q)}")

    def reset_stats(self):
        pass

    def get_q_value(self, state, action):
        if self.canonical:
            state, index_mapping = NimLogic.canonicalize_state(state)
            action = (index_mapping[action[0]], action[1])

        return self.q.get((tuple(state), action), 0.0)

    def set_q_value(self, state, action, value):
        if self.canonical:
            state, index_mapping = NimLogic.canonicalize_state(state)
            action = (index_mapping[action[0]], action[1])

        self.q[(tuple(state), action)] = value

    def get_state_value(self, state):
        actions = NimLogic.available_actions(state)

        if not actions:
            return 0.0

        return max(self.get_q_value(state, a) for a in actions)

    def choose_action(self, piles, training=False):
        actions = list(NimLogic.available_actions(piles))
        if not actions:
            return None

        if training and random.random() < self.epsilon:
            return random.choice(actions)

        best_value = float('-inf')
        best_actions = []

        for action in actions:
            value = self.get_q_value(piles, action)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def learn_from_transition(self, state, action, next_state, game_over):
        current_q = self.get_q_value(state, action)

        if game_over:
            if self.misere:
                future_value = -1.0
            else:
                future_value = 1.0
        else:
            future_value = -self.get_state_value(next_state)

        new_q = current_q + self.alpha * (future_value - current_q)
        self.set_q_value(state, action, new_q)

    def train(self, num_episodes):
        for _ in tqdm(range(num_episodes)):
            piles = [self.max_pile] * self.pile_count

            state = NimGameState(piles, self.misere)

            while state.winner is None:
                current_piles = state.piles.copy()

                action = self.choose_action(current_piles, training=True)
                if action is None:
                    break

                state = state.apply_move(action)

                game_over = state.winner is not None
                self.learn_from_transition(
                    current_piles,
                    action,
                    state.piles.copy(),
                    game_over
                )

    def _save(self):
        save_dict = {
            'q_table': {
                f"{','.join(map(str, k[0]))}|{k[1][0]},{k[1][1]}": v
                for k, v in self.q.items()
            },
            'metadata': {
                'pile_count': self.pile_count,
                'max_pile': self.max_pile,
                'misere': self.misere,
                'q_size': len(self.q),
                'alpha': self.alpha,
                'epsilon': self.epsilon,
                'gamma': self.gamma,
                'canonical': self.canonical
            }
        }

        with open(self.save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        print(f"Saved agent to {self.save_path}")

    def _load(self):
        if not os.path.exists(self.save_path):
            return False

        try:
            with open(self.save_path, 'r') as f:
                save_dict = json.load(f)

            self.q = {}
            for key_str, value in save_dict['q_table'].items():
                state_str, action_str = key_str.split('|')
                state = tuple(map(int, state_str.split(',')))
                action = tuple(map(int, action_str.split(',')))
                self.q[(state, action)] = value

            print(f"Loaded agent from {self.save_path}")
            return True

        except Exception as e:
            print(f"Error loading agent: {e}")
            return False