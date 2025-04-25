from agents.Agent import Agent


class HumanAgent(Agent):
    def __init__(self):
        super().__init__("Human")

    def choose_action(self, piles):
        print("Your Turn")
        available_actions = self.get_available_actions(piles)

        while True:
            pile = self.get_int("Choose Pile: ")
            count = self.get_int("Choose Count: ")
            if (pile, count) in available_actions:
                break
            print("Invalid move, try again.")

        return pile, count

    def get_available_actions(self, piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    def get_int(self, prompt):
        while True:
            user_input = input(prompt)
            try:
                return int(user_input)
            except ValueError:
                continue