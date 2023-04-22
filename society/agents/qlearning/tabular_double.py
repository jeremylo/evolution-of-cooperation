from random import choice, random
from typing import List

import numpy as np

from society.action import Action
from society.agent import Agent


class DoubleTabularQLearner(Agent):
    def __init__(
        self,
        lookback: int = 1,
        epsilon: float = 0.1,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
    ) -> None:
        super().__init__()

        self._lookback = lookback
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._q_table1 = np.zeros(shape=tuple(4 for _ in range(self._lookback)) + (2,))
        self._q_table2 = np.zeros(shape=tuple(4 for _ in range(self._lookback)) + (2,))

    @property
    def parameters(self) -> dict:
        return {
            "lookback": self._lookback,
            "epsilon": self._epsilon,
            "learning_rate": self._learning_rate,
            "discount_factor": self._discount_factor,
        }

    @property
    def _q_table(self):
        return self._q_table1 + self._q_table2

    def _to_state(self, history: List[Action], opp_history: List[Action]) -> tuple:
        state = tuple(
            2 * a.value + b.value
            for a, b in zip(history[-self._lookback :], opp_history[-self._lookback :])
        )

        if len(state) < self._lookback:
            state = (0,) * (self._lookback - len(state)) + state

        return state

    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        # Play a random move
        if self.training and random() < self._epsilon:
            return choice((Action.COOPERATE, Action.DEFECT))

        # Return the action corresponding to
        # the largest Q-value for this state
        return (Action.COOPERATE, Action.DEFECT)[
            self._q_table[self._to_state(history, opp_history)].argmax()
        ]

    def update(self, reward: int, history: List[Action], opp_history: List[Action]):
        if len(history) <= self._lookback:
            return

        move1 = history[-1].value

        old_state = self._to_state(history[:-1], opp_history[:-1])
        new_state = self._to_state(history, opp_history)

        if random() < 0.5:
            action = self._q_table1[new_state].argmax()
            self._q_table1[old_state][move1] += self._learning_rate * (
                reward
                + self._discount_factor * self._q_table2[new_state][action]
                - self._q_table1[old_state][move1]
            )
        else:
            action = self._q_table2[new_state].argmax()
            self._q_table2[old_state][move1] += self._learning_rate * (
                reward
                + self._discount_factor * self._q_table1[new_state][action]
                - self._q_table2[old_state][move1]
            )

    def save_q_table(self, file: str):
        """Saves the current Q-table to a file.

        Args:
            file (str): A file path.
        """

        np.savez(file, q_table1=self._q_table1, q_table2=self._q_table2)
