import json, os
import numpy as np
from tqdm import tqdm

from nim.NimLogic import NimLogic
from nim.NimGameState import NimGameState

from helpers.HelperLogic import HelperLogic

from base.BaseAgent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    An agent that uses Q-learning to play Nim.
    """
    def __init__(self, misere, pile_count, max_pile,
                 num_episodes, override=False,
                 alpha=0.3, epsilon=0.3, gamma=1.0,
                 canonical=False, reduced=False):
        """
        Initializes the QLearning agent.
        """
        super().__init__()

        self.misere: bool = misere
        self.pile_count: int = pile_count
        self.max_pile: int = max_pile

        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q: dict = dict()

        self.canonical: bool = canonical or reduced
        self.reduced: bool = reduced

        self.save_dir: str = "../savedAgents/QLearning"
        os.makedirs(self.save_dir, exist_ok=True)

        self.filename: str = (f"qlearning-{pile_count}-{max_pile}-{'misere' if misere else 'normal'}"
                         f"{'-canonical' if canonical else ''}"
                         f"{'-reduced' if reduced else ''}-{num_episodes}.json")
        self.save_path: str = os.path.join(self.save_dir, self.filename)

        if not self._load() and not override:
            self._train(num_episodes)
            self._save()

        print(f"{self} ready. Q-table size: {len(self.q)}")

    def __str__(self) -> str:
        """
        Returns a string representation of the agent.
        """
        return f"{'Reduced ' if self.reduced else 'Canonical ' if self.canonical else ''}QLearning Agent"

    def reset_stats(self):
        """
        Resets the statistics of the agent.
        """
        pass

    def get_stats(self) -> tuple | None:
        """
        Returns the statistics of the agent.
        """
        return None

    def choose_action(self, state: list[int], training: bool = False) -> tuple[int, int]:
        """
        Chooses an action based on the current state of the game.
        """
        index_mapping: list[int]
        current_piles: list[int]
        current_piles, index_mapping = self._preprocess_state(state)

        actions: set[tuple[int, int]] = NimLogic.available_actions(current_piles)
        action: tuple[int, int] = self._choose_action(current_piles, actions, training)
        action = self._postprocess_action(action, index_mapping)

        return action

    def _get_q_value(self, state: list[int], action: tuple[int, int]) -> float:
        """
        Retrieves the Q-value for a given state-action pair.
        """
        return self.q.get((tuple(state), action), 0.0)

    def _set_q_value(self, state: list[int], action: tuple[int, int], value: float) -> None:
        """
        Sets the Q-value for a given state-action pair.
        """
        self.q[(tuple(state), action)] = value

    def _get_state_value(self, state: list[int]):
        """
        Computes the maximum Q-value for all available actions in the given state.
        """
        actions: set[tuple[int, int]] = NimLogic.available_actions(state)
        return max((self._get_q_value(state, a) for a in actions), default=0.0)

    def _choose_action(self, state: list[int], actions: set[tuple[int, int]], training: bool = False) -> tuple[int, int]:
        """
        Chooses an action based on the current state and available actions.
        If training and epsilon-greedy condition is met, a random action is chosen.
        """
        if training and np.random.random() < self.epsilon:
            return NimLogic.random_action(actions)

        q_vals: list[tuple[float, tuple[int, int]]] = [(self._get_q_value(state, a), a) for a in actions]
        max_q: float = max(q_vals, key=lambda x: x[0])[0]
        best: list[tuple[int, int]] = [a for q, a in q_vals if q == max_q]

        idx: int = np.random.randint(len(best))
        return best[idx]

    def _learn_from_transition(self, state: list[int], action: tuple[int, int], next_state: list[int], game_over: bool) -> None:
        """
        Updates the Q-value based on the transition from the current state to the next state.
        """
        current_q: float = self._get_q_value(state, action)
        future_value: float

        if game_over:
            future_value = -1.0 if self.misere else 1.0
        else:
            future_value = -self.gamma * self._get_state_value(next_state)

        new_q: float  = current_q + self.alpha * (future_value - current_q)
        self._set_q_value(state, action, new_q)

    def _train(self, num_episodes):
        """
        Trains the agent by simulating a number of episodes.
        """
        for _ in tqdm(range(num_episodes)):
            game_state: NimGameState = NimGameState([self.max_pile] * self.pile_count, self.misere)

            while game_state.winner is None:
                current_piles: list[int] = game_state.piles.copy()
                index_mapping: list[int]

                current_piles, index_mapping = self._preprocess_state(current_piles)

                actions: set[tuple[int, int]] = NimLogic.available_actions(current_piles)
                action: tuple[int, int] = self._choose_action(current_piles, actions, training=True)

                game_state: NimGameState = game_state.apply_move(
                    self._postprocess_action(action, index_mapping)
                )

                new_piles: list[int] = game_state.piles.copy()
                new_piles, _ = self._preprocess_state(new_piles)

                self._learn_from_transition(
                    current_piles,
                    action,
                    new_piles,
                    game_state.winner is not None
                )

    def _preprocess_state(self, current_piles: list[int]) -> tuple[list[int], list[int]]:
        """
        Preprocesses the current state of the game to prepare for action selection.
        """
        index_mapping: list[int] = list()
        current_piles: list[int] = current_piles.copy()

        if self.canonical:
            current_piles, index_mapping = HelperLogic.canonicalize_state(current_piles)

        if self.reduced:
            current_piles = HelperLogic.reduce_state(current_piles)

        return current_piles, index_mapping

    def _postprocess_action(self, action: tuple[int, int], index_mapping: list[int]) -> tuple[int, int]:
        """
        Post-processes the action after it has been chosen.
        """
        if self.canonical:
            return HelperLogic.map_action_to_original(action, index_mapping)

        return action


    def _save(self):
        """
        Saves the agent's Q-table and metadata to a JSON file.
        """
        save_dict: dict[str, object] = {
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
        """
        Loads the agent's Q-table and metadata from a JSON file.
        """
        if not os.path.exists(self.save_path):
            return False

        try:
            with open(self.save_path, 'r') as f:
                save_dict: dict = json.load(f)

            self.q: dict[tuple[tuple[int, ...], tuple[int, int]], float] = dict()

            for key_str, value in save_dict['q_table'].items():
                state_str: str
                action_str: str
                state_str, action_str = key_str.split('|')

                state: tuple[int, ...] = tuple(map(int, state_str.split(',')))
                action: tuple[int, int] = tuple[int, int](map(int, action_str.split(',')))

                self.q[(state, action)] = value

            print(f"Loaded agent from {self.save_path}")
            return True

        except Exception as e:
            print(f"Error loading agent: {e}")
            return False