import numpy as np

from nim.NimGame import NimGame

from agents.HumanAgent import HumanAgent
from agents.AlgorithmicAgent import AlgorithmicAgent

from agents.MinimaxAgent import MinimaxAgent
from agents.QLearningAgent import QLearningAgent

MAX_PILE = 7
PILE_COUNT = 4
MISERE = np.random.rand() < 0.5

agent1 = HumanAgent()
agent2 = AlgorithmicAgent(misere=MISERE)

if np.random.rand() < 0.5:
    agent1, agent2 = agent2, agent1

game = NimGame(
    initial_piles=np.random.randint(1, MAX_PILE, size=PILE_COUNT),
    misere=MISERE
)

game.play(
    player1=agent1,
    player2=agent2,
    verbose=True
)