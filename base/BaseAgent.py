class BaseAgent:
    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError

    def reset_stats(self):
        raise NotImplementedError

    def choose_action(self, state: list[int]) -> tuple[int, int]:
        raise NotImplementedError