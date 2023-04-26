from typing import List

from society.action import Action
from society.agent import Agent


class TitForTat(Agent):
    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        # Cooperate initially
        if not opp_history:
            return Action.COOPERATE

        # Play the opponent's previous action
        return opp_history[-1]
