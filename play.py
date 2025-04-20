from nim.Nim import *

from agents.otherAgents.HumanAgent import HumanAgent


def play(player1, player2, initial_piles, misere, verbose=True):
    game = Nim(initial=initial_piles, misere=misere)
    players = [player1, player2]
    player_names = ["Player 1", "Player 2"]

    while True:
        if verbose:
            print("\nPiles:")
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
                print(f"\nGAME OVER\nWinner is {player_names[game.winner]} ({players[game.winner].name})")
            break

    return game.winner


if __name__ == '__main__':
    initial_piles = [5, 6, 7, 8, 9, 10]

    player1 = HumanAgent()
    player2 = HumanAgent()

    misere = False

    play(
        player1=player1,
        player2=player2,
        initial_piles=initial_piles,
        misere=misere,
        verbose=True
    )