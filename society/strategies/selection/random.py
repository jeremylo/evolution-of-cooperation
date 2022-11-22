from random import randrange
from typing import List

from society.strategy import SelectionStrategy


class RandomSelectionStrategy(SelectionStrategy):
    def select_partner(self, returns: List[List[float]]) -> int:
        return randrange(len(returns))
