from nim import NimLogic

from agents import Agent


class HumanAgent(Agent):
    def __init__(self):
        super().__init__()

    def reset_stats(self):
        return

    def choose_action(self, piles):
        print("Your Turn")
        available_actions = NimLogic.available_actions(piles)

        while True:
            pile = self.get_int("Choose Pile: ")
            count = self.get_int("Choose Count: ")
            if (pile, count) in available_actions:
                break
            print("Invalid move, try again.")

        return pile, count

    @staticmethod
    def get_int(prompt):
        while True:
            user_input = input(prompt)
            try:
                return int(user_input)
            except ValueError:
                continue