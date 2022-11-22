from typing import List

import numpy as np

from society.strategy import GameplayStrategy, SelectionStrategy


class Agent:

    selection_strategy: SelectionStrategy
    gameplay_strategy: GameplayStrategy

    def __init__(
        self,
        selection_strategy: SelectionStrategy,
        gameplay_strategy: GameplayStrategy,
        index: int = None,
        population: int = None,
    ) -> None:
        self.selection_strategy = selection_strategy
        self.gameplay_strategy = gameplay_strategy
        self.index = index
        self.population = population

    def set_index(self, index: int, population: int):
        self.index = index
        self.population = population

    def select_partner(self, state: List[float]) -> int:
        partner = self.selection_strategy.select_partner(np.roll(state, -self.index))

        assert partner in range(self.population)

        return (partner + self.index) % self.population


class TrainableAgent(Agent):
    def update_selector(
        self, old_state: List[List[float]], new_state: List[List[float]], reward: float
    ):
        self.selection_strategy.update(old_state, new_state, reward)
