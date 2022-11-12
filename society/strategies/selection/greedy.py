from random import random, randrange
from typing import Dict, List

import numpy as np

from society.strategy import SelectionStrategy


class GreedySelectionStrategy(SelectionStrategy):
    def select_partner(self, returns: Dict[int, float]) -> int:
        return np.argmax([sum(r) for r in returns])


class EGreedySelectionStrategy(SelectionStrategy):
    def __init__(self, epsilon: float = 0.1) -> None:
        super().__init__()
        self.epsilon = epsilon

    def select_partner(self, returns: List[List[float]]) -> int:
        if random() < self.epsilon:
            return randrange(len(returns))

        return np.argmax([sum(r) for r in returns])
