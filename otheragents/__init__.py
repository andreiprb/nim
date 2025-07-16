"""
This module initializes the other agents package, which contains the following:
- HumanAgent: An agent that allows users to play against the computer.
- MathematicalAgent: An agent that plays perfectly based on the Nim-sum.
"""

from .HumanAgent import HumanAgent
from .MathematicalAgent import MathematicalAgent

__all__ = ['HumanAgent', 'MathematicalAgent']