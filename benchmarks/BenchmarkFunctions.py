from tqdm import tqdm

import numpy as np

from nim.NimGame import NimGame
from nim.NimLogic import NimLogic

from base.BaseAgent import BaseAgent

from agents.AlgorithmicAgent import AlgorithmicAgent


class BenchmarkFunctions:
    @staticmethod
    def test_agent(misere_agent: BaseAgent, normal_agent: BaseAgent,
                   initial_piles: np.ndarray, game_modes: np.ndarray,
                   episodes: int) -> tuple | None:
        """
        Tests the given agents by playing a series of Nim games with random initial piles and game modes.
        """
        for i in tqdm(range(episodes)):
            game: NimGame = NimGame(
                initial_piles=list[int](initial_piles[i].tolist()),
                misere=bool(game_modes[i])
            )

            agent1: BaseAgent = misere_agent if game_modes[i] else normal_agent
            agent2: BaseAgent = AlgorithmicAgent(misere=game_modes[i])

            winner: int = game.play(
                player1=agent1,
                player2=agent2,
            )

            """ AGENT VALIDATION """
            assert winner == NimLogic.is_p_position(list[int](initial_piles[i]), bool(game_modes[i])), "Bad agent!"

    @staticmethod
    def run_tests(misere_agents: dict[str, BaseAgent], normal_agents: dict[str, BaseAgent],
                  pile_count: int, max_pile: int, episodes: int):
        """
        Runs a series of tests for the given agents by simulating Nim games with random initial piles and game modes.
        """
        print("-" * 60)
        print(f"Configuration: pile_count: {pile_count}, max_pile: {max_pile}")
        print("-" * 60)

        initial_piles: np.ndarray = np.random.randint(1, max_pile + 1, size=(episodes, pile_count))
        game_modes: np.ndarray = np.random.choice([False, True], size=episodes)

        for agent_key in misere_agents.keys():
            misere_agent: BaseAgent = misere_agents[agent_key]
            normal_agent: BaseAgent = normal_agents[agent_key]

            print(f"Testing {misere_agent}")

            BenchmarkFunctions.test_agent(
                misere_agent=misere_agent,
                normal_agent=normal_agent,
                initial_piles=initial_piles,
                game_modes=game_modes,
                episodes=episodes
            )