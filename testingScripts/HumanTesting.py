import numpy as np

from Nim.Nim import Nim

from Agents.HumanAgent import HumanAgent
from Agents.MathAgent import MathAgent

from Agents.MinimaxAgent import MinimaxAgent
from Agents.QLearningAgent import QLearningAgent

MAX_PILE = 7
PILE_COUNT = 4
MISERE = np.random.rand() < 0.5

agent1 = HumanAgent()
agent2 = MathAgent(misere=MISERE)

if np.random.rand() < 0.5:
    agent1, agent2 = agent2, agent1

game = Nim(
    initial_piles=np.random.randint(1, MAX_PILE, size=PILE_COUNT),
    misere=MISERE
)

game.play(
    player1=agent1,
    player2=agent2,
    verbose=True
)