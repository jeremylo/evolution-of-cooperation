from random import choice
from typing import List

from society.action import Action
from society.agent import Agent


class Random(Agent):
    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        return choice([Action.COOPERATE, Action.DEFECT])
