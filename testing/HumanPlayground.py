import numpy as np

from Nim.Nim import Nim

from Agents.HumanAgent import HumanAgent

from Agents.MinimaxAgent import MinimaxAgent
from Agents.QLearningAgent import QLearningAgent

MAX_PILE = 7
PILE_COUNT = 4
MISERE = True

agent1 = HumanAgent()
agent2 = QLearningAgent(misere=MISERE, pile_count=PILE_COUNT, max_pile=MAX_PILE, canonical=True, num_episodes=1000000)

if np.random.rand() < 0.5:
    agent1, agent2 = agent2, agent1

game = Nim(
    initial_piles=np.random.randint(1, MAX_PILE, size=PILE_COUNT),
    misere=np.random.rand() < 0.5
)

game.play(
    player1=agent1,
    player2=agent2,
    verbose=True
)