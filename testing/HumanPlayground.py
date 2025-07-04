import numpy as np

from Nim.Nim import Nim

from Agents.HumanAgent import HumanAgent

from Agents.MinimaxAgent import MinimaxAgent

MAX_PILE = 5
PILE_COUNT = 4
MISERE = True

agent1 = HumanAgent()
agent2 = MinimaxAgent(misere=MISERE, max_depth=5)

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