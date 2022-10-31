from typing import List

from society.action import Action
from society.strategy import GameplayStrategy


class AllC(GameplayStrategy):
    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        return Action.COOPERATE


class AllD(GameplayStrategy):
    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        return Action.DEFECT
