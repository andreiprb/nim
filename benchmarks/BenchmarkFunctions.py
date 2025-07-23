import os
from tqdm import tqdm
from typing import Callable

import numpy as np

from nim.NimGame import NimGame
from nim.NimLogic import NimLogic

from base.BaseAgent import BaseAgent

from otheragents.MathematicalAgent import MathematicalAgent


class BenchmarkFunctions:
    @staticmethod
    def benchmark_agent(misere_agent: BaseAgent, normal_agent: BaseAgent,
                        initial_piles: np.ndarray, game_modes: np.ndarray,
                        episodes: int) -> np.ndarray | None:
        """
        Tests the given agents by playing a series of nim games with random
        initial piles and game modes.
        Each game is played against an algorithmic agent that uses the known
        mathematical strategy.
        Each game is also checked against the expected outcome based on the
        nim-sum logic.
        """
        statistics: np.ndarray = np.full(episodes, None, dtype=object)
        has_data: bool = False

        for episode in tqdm(range(episodes)):
            game: NimGame = NimGame(
                initial_piles=list[int](initial_piles[episode].tolist()),
                misere=bool(game_modes[episode])
            )

            agent1: BaseAgent = \
                misere_agent if game_modes[episode] else normal_agent
            agent2: BaseAgent = MathematicalAgent(misere=game_modes[episode])

            winner: int = game.play(
                player1=agent1,
                player2=agent2,
            )

            assert winner == NimLogic.is_p_position(
                list[int](initial_piles[episode]),
                bool(game_modes[episode])
            ), "Bad agent!"

            stats = agent1.get_stats()
            if stats is not None:
                has_data = True
                statistics[episode] = stats

        return statistics if has_data else None

    @staticmethod
    def run_benchmarks(misere_agents: dict[str, BaseAgent],
                       normal_agents: dict[str, BaseAgent],
                       heap_count: int, max_heap: int, episodes: int,
                       processing: Callable[[np.ndarray], None] | None,
                       save_path_prefix: str = "") -> None:
        """
        Runs a series of tests for the given agents by simulating nim games
        with random initial piles and game modes.
        """
        print("-" * 60)
        print(f"Configuration: pile_count: {heap_count}, max_pile: {max_heap}")
        print("-" * 60)

        initial_piles: np.ndarray = np.random.randint(
            1, max_heap + 1, size=(episodes, heap_count))
        game_modes: np.ndarray = np.random.choice(
            [False, True], size=episodes)

        for agent_key in misere_agents.keys():
            if save_path_prefix:
                agent_name: str = save_path_prefix.split("/")[-1].lower()
                filename: str = (f"{save_path_prefix}/"
                                 f"{agent_name}-{agent_key}-"
                                 f"{heap_count}-{max_heap}-{episodes}.npz")

                if os.path.exists(filename):
                    print(f"Skipping {filename}")
                    continue

            misere_agent: BaseAgent = misere_agents[agent_key]
            normal_agent: BaseAgent = normal_agents[agent_key]

            print(f"Testing {misere_agent}")

            statistics = BenchmarkFunctions.benchmark_agent(
                misere_agent=misere_agent,
                normal_agent=normal_agent,
                initial_piles=initial_piles,
                game_modes=game_modes,
                episodes=episodes
            )

            if statistics is None or processing is None:
                continue

            processing(statistics)