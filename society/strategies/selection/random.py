from random import randrange
from typing import Dict, List

from society.strategy import SelectionStrategy


class RandomSelectionStrategy(SelectionStrategy):
    def select_partner(self, returns: Dict[int, List[float]]) -> int:
        return randrange(len(returns))
