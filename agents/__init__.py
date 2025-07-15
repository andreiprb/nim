"""
This module initializes the agent package, importing various agent classes
"""

from .HumanAgent import HumanAgent

from .AlgorithmicAgent import AlgorithmicAgent

from .MinimaxAgent import MinimaxAgent
from .QLearningAgent import QLearningAgent


__all__ = ['HumanAgent', 'AlgorithmicAgent','MinimaxAgent', 'QLearningAgent']