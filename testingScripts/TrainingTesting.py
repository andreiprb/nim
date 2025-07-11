from Nim.Nim import Nim
from Nim.NimLogic import NimLogic

from Agents.QLearningAgent import QLearningAgent


PILES = [2, 7, 6, 5]
MISERE = False

PILE_COUNT = 4
MAX_PILE = 7


game = Nim(
    initial_piles=PILES,
    misere=MISERE,
)

agent = QLearningAgent(misere=True, pile_count=PILE_COUNT, max_pile=MAX_PILE, num_episodes=1000, reduced=True)


winner = game.play(
    player1=agent,
    player2=agent,
    verbose=True
)

assert winner == NimLogic.is_p_position(PILES, MISERE), "Bad agent!"
