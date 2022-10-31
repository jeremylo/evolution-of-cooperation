from random import choice, random
from typing import Dict

from society.strategy import SelectionStrategy


class GreedySelectionStrategy(SelectionStrategy):
    def select_partner(self, returns: Dict[int, float]) -> int:
        return max(returns, key=lambda i: sum(returns[i]))


class EGreedySelectionStrategy(SelectionStrategy):
    def __init__(self, epsilon: float = 0.1) -> None:
        super().__init__()
        self.epsilon = epsilon

    def select_partner(self, returns: Dict[int, float]) -> int:
        if random() < self.epsilon:
            return choice(list(returns.keys()))

        return max(returns, key=lambda i: sum(returns[i]))
