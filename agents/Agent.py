class Agent:
    def __init__(self, name):
        self.name = name + " agent"

    def train(self):
        if "minimax" in self.name or "Nim" in self.name:
            print(f"Playing {self.name}. No need for training.")
            return

        print(f"Training {self.name}...")
