import numpy as np

from nim import NimGame

from base import BaseAgent

from otheragents import HumanAgent
from otheragents import MathematicalAgent

MAX_PILE: int = 7
PILE_COUNT: int = 4
MISERE: bool = np.random.rand() < 0.5

agent1: BaseAgent = HumanAgent()
agent2: BaseAgent = MathematicalAgent(misere=MISERE)

if np.random.rand() < 0.5:
    agent1, agent2 = agent2, agent1

game = NimGame(
    initial_piles=list[int](np.random.randint(
        1,
        MAX_PILE + 1,
        size=PILE_COUNT
    ).tolist()),
    misere=MISERE
)

game.play(
    player1=agent1,
    player2=agent2,
    verbose=True
)