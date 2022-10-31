from typing import Dict

from society.strategy import GameplayStrategy, SelectionStrategy


class Agent:

    selection_strategy: SelectionStrategy
    gameplay_strategy: GameplayStrategy

    def __init__(
        self, selection_strategy: SelectionStrategy, gameplay_strategy: GameplayStrategy
    ) -> None:
        self.selection_strategy = selection_strategy
        self.gameplay_strategy = gameplay_strategy

    def select_partner(self, returns: Dict[int, float]) -> int:
        return self.selection_strategy.select_partner(returns)


class TrainableAgent:
    pass
