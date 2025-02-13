import os
from openai import OpenAI
from dotenv import load_dotenv

from nim.NimLogic import NimLogic


class LLMAgent:
    def __init__(self, model="gpt-4o-mini", temperature=0.0):
        load_dotenv()

        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

        self.system_prompt = """You are playing the misere version of Nim. Here are the rules:
        1. There are multiple piles of objects
        2. On your turn, you must remove one or more objects from a single pile
        3. The player who takes the last object loses
        4. You must respond with only two numbers: the pile index (0-based) and the number of objects to remove
        5. Your response should be in the format: pile_index,count

        Your goal is to make optimal moves to win the game. Think through the position carefully before making your move."""

    def _format_state(self, state):
        return f"""Current game state (pile indices and their sizes):
        {', '.join(f'Pile {i}: {count}' for i, count in enumerate(state))}

        What is your move? Respond with only two numbers separated by a comma:
        pile_index,count"""

    def _parse_response(self, response_text, state):
        try:
            clean_response = response_text.strip().split('\n')[0]
            pile, count = map(int, clean_response.split(','))
            return (pile, count)
        except (ValueError, IndexError):
            return self._get_fallback_move(state)

    def _get_fallback_move(self, state):
        valid_actions = NimLogic.available_actions(state)
        return next(iter(valid_actions))

    def _is_valid_move(self, move, state):
        pile, count = move
        if pile < 0 or pile >= len(state):
            return False
        if count < 1 or count > state[pile]:
            return False
        return True

    def choose_action(self, state, epsilon=None):
        prompt = self._format_state(state)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )

            move = self._parse_response(response.choices[0].message.content, state)

            if self._is_valid_move(move, state):
                return move
            else:
                return self._get_fallback_move(state)

        except Exception as e:
            print(f"Error getting move from {self.model}: {e}")
            return self._get_fallback_move(state)

    def train(self):
        print(f"Playing versus Large Language Model agent ({self.model}). Nothing to train.")