from random import choice, random
from typing import List, Optional

import numpy as np

from society.action import Action
from society.strategy import GameplayStrategy

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99


class TabularQLearningGameplayStrategy(GameplayStrategy):
    def __init__(self, epsilon: float = 0.1, file: Optional[str] = None) -> None:
        super().__init__()

        self._epsilon = epsilon
        self._q_table = np.load(file)["q_table"] if file else np.zeros((2, 2, 2))

    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        # Cooperate initially
        if not history:
            return Action.COOPERATE

        # Play a random move with probability epsilon
        if random() < self._epsilon:
            return choice((Action.COOPERATE, Action.DEFECT))

        # Return the action corresponding to
        # the largest Q-value for this state
        return (Action.COOPERATE, Action.DEFECT)[
            self._q_table[history[-1].value, opp_history[-1].value].argmax()
        ]

    def update(self, reward: int, history: List[Action], opp_history: List[Action]):
        if len(history) <= 2:
            return

        move1, move2 = history[-1].value, opp_history[-1].value

        old_state = history[-2].value, opp_history[-2].value
        new_state = (move1, move2)

        self._q_table[old_state][move1] += LEARNING_RATE * (
            reward
            + DISCOUNT_FACTOR * self._q_table[new_state].max()
            - self._q_table[old_state][move1]
        )

    def save_q_table(self, file: str):
        """Saves the current Q-table to a file.

        Args:
            file (str): A file path.
        """

        np.savez(file, q_table=self._q_table)
