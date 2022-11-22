from random import random, randrange
from typing import Dict, List

import numpy as np

from society.strategy import SelectionStrategy


class GreedySelectionStrategy(SelectionStrategy):
    def select_partner(self, returns: List[List[float]]) -> int:
        try:
            return np.mean(returns, axis=1, dtype=object).argmax()
        except:
            return randrange(len(returns))


class EGreedySelectionStrategy(SelectionStrategy):
    def __init__(self, epsilon: float = 0.1) -> None:
        super().__init__()
        self.epsilon = epsilon

    def select_partner(self, returns: List[List[float]]) -> int:
        if random() < self.epsilon:
            return randrange(len(returns))

        try:
            return np.mean(returns, axis=1, dtype=object).argmax()
        except:
            return randrange(len(returns))
