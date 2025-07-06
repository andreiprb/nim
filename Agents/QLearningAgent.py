import random
import json
import os
import numpy as np
from tqdm import tqdm

from Nim.NimLogic import NimLogic
from Nim.NimGameState import NimGameState


class QLearningAgent:
    def __init__(self, misere, pile_count, max_pile,
                 num_episodes, override=False,
                 alpha=0.3, epsilon=0.3, gamma=1.0,
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
        self.filename = f"qlearning-{pile_count}-{max_pile}-{'misere' if misere else 'normal'}-{num_episodes}{'-canonical' if canonical else ''}.json"
        self.save_path = os.path.join(self.save_dir, self.filename)

        if not self._load() and not override:
            self.train(num_episodes)
            self._save()

        print(f"Agent ready. Q-table size: {len(self.q)}")

    def reset_stats(self):
        pass

    def get_q_value(self, state, action):
        return self.q.get((tuple(state), action), 0.0)

    def set_q_value(self, state, action, value):
        self.q[(tuple(state), action)] = value

    def get_state_value(self, state):
        actions = NimLogic.available_actions(state)

        if not actions:
            return 0.0

        return max(self.get_q_value(state, a) for a in actions)

    def choose_action(self, state, training=False):
        if self.canonical:
            state, index_mapping = NimLogic.canonicalize_state(state)

        actions = list(NimLogic.available_actions(state))

        if not actions:
            return None

        chosen_action = self._choose_action(state, actions, training)

        if self.canonical:
            chosen_action = NimLogic.map_action_to_original(chosen_action, index_mapping)

        return chosen_action

    def _choose_action(self, state, actions, training=False):
        if training and np.random.random() < self.epsilon:
            idx = np.random.randint(len(actions))
            return actions[idx]

        q_vals = [(self.get_q_value(state, a), a) for a in actions]
        max_q = max(q_vals, key=lambda x: x[0])[0]
        best = [a for q, a in q_vals if q == max_q]

        idx = np.random.randint(len(best))
        return best[idx]

    def learn_from_transition(self, state, action, next_state, game_over):
        current_q = self.get_q_value(state, action)

        if game_over:
            future_value = -1.0 if self.misere else 1.0
        else:
            future_value = -self.gamma * self.get_state_value(next_state)

        new_q = current_q + self.alpha * (future_value - current_q)
        self.set_q_value(state, action, new_q)

    def train(self, num_episodes):
        for _ in tqdm(range(num_episodes)):
            game_state = NimGameState([self.max_pile] * self.pile_count, self.misere)
            while game_state.winner is None:
                current_piles = game_state.piles.copy()

                if self.canonical:
                    current_piles, index_mapping = NimLogic.canonicalize_state(current_piles)

                actions = list(NimLogic.available_actions(current_piles))

                if not actions:
                    break

                action = self._choose_action(current_piles, actions, training=True)
                copy_action = action if not self.canonical else NimLogic.map_action_to_original(action, index_mapping)

                game_state = game_state.apply_move(copy_action)
                new_state_piles = game_state.piles if not self.canonical else NimLogic.canonicalize_state(game_state.piles)[0]

                self.learn_from_transition(
                    current_piles,
                    action,
                    new_state_piles.copy(),
                    game_state.winner is not None
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