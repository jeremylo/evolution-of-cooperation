from typing import List

from society.action import Action
from society.agent import Agent


class Pavlov(Agent):
    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        if not opp_history or history[-1] == opp_history[-1]:
            return Action.COOPERATE

        return Action.DEFECT
