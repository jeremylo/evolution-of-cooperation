from random import choice
from typing import Dict

from society.strategy import SelectionStrategy


class RandomSelectionStrategy(SelectionStrategy):
    def select_partner(self, returns: Dict[int, float]) -> int:
        return choice(list(returns.keys()))
