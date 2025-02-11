from agents.QLearningAgent import QLearningAgent
from agents.AlphaZeroAgent import AlphaZeroAgent
from agents.MinimaxAgent import MinimaxAgent

from nim.NimLogic import NimLogic


AGENT = AlphaZeroAgent


class Nim:

    def __init__(self, initial=[1, 3, 5, 7]):
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    def switch_player(self):
        self.player = NimLogic.other_player(self.player)

    def move(self, action):
        pile, count = action

        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        self.piles[pile] -= count
        self.switch_player()

        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


def train(n):
    player = AGENT()

    if AGENT is MinimaxAgent:
        print("Playing versus Minimax agent")
        return player

    else:
        print(f"Training Q-Learning agent for {n} episodes...")
        for i in range(n):
            game = Nim()

            last = {
                0: {"state": None, "action": None},
                1: {"state": None, "action": None}
            }

            while True:

                state = game.piles.copy()
                action = player.choose_action(game.piles)

                last[game.player]["state"] = state
                last[game.player]["action"] = action

                game.move(action)
                new_state = game.piles.copy()

                if game.winner is not None:
                    player.update(state, action, new_state, -1)
                    player.update(
                        last[game.player]["state"],
                        last[game.player]["action"],
                        new_state,
                        1
                    )
                    break

                elif last[game.player]["state"] is not None:
                    player.update(
                        last[game.player]["state"],
                        last[game.player]["action"],
                        new_state,
                        0
                    )

        print("Done training")

        return player
