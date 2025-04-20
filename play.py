from nim.Nim import *
from nim.NimLogic import NimLogic

from agents.algorithmicAgents.MinimaxAgent import MinimaxAgent
from agents.algorithmicAgents.MinimaxNimAgent import MinimaxNimAgent


def play_game(p1, p2, piles, misere, verbose=True):
    game = Nim(initial=piles, misere=misere)
    players = [p1, p2]
    player_names = ["Player 1", "Player 2"]

    while True:
        if verbose:
            print()
            print("Piles:")
            for i, pile in enumerate(game.piles):
                print(f"Pile {i}: {pile}")
            print()

        current_player = players[game.player]
        if verbose:
            print(f"{player_names[game.player]}'s Turn ({current_player.name})")

        pile, count = current_player.choose_action(game.piles)
        if verbose:
            print(f"{player_names[game.player]} chose to take {count} from pile {pile}.")

        game.move((pile, count))

        if game.winner is not None:
            if verbose:
                print()
                print("GAME OVER")
                print(f"Winner is {player_names[game.winner]} ({players[game.winner].name})")
            break

    return game.winner


if __name__ == '__main__':
    initial_piles = [5, 6, 7, 8, 9, 10]

    player1 = MinimaxAgent()
    player2 = MinimaxNimAgent()

    wins = [0, 0]

    for _ in range(10000):
        winner = play_game(
            p1=player1,
            p2=player2,
            piles=initial_piles,
            misere=False,
            verbose=False
        )

        wins[winner] += 1

    print(f"{wins[0]} - {wins[1]}")