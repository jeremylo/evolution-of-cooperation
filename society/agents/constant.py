from typing import List

from society.action import Action
from society.agent import Agent


class AllC(Agent):
    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        return Action.COOPERATE


class AllD(Agent):
    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        return Action.DEFECT
