from nim.Nim import *
from nim.NimLogic import NimLogic

from agents.algorithmicAgents.MinimaxAgent import MinimaxAgent
from agents.algorithmicAgents.MinimaxNimAgent import MinimaxNimAgent


def play_game(player1, player2, initial_piles, verbose=True):
    game = Nim(initial=initial_piles)
    players = [player1, player2]
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

    player1wins = 0
    player2wins = 0

    for _ in range(10000):
        winner = play_game(player1, player2, initial_piles, False)

        if winner:
            player2wins += 1

        else:
            player1wins += 1

    print(player1wins, '-', player2wins)