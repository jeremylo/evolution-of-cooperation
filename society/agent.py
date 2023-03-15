from typing import List

from society.action import Action


class Agent:
    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        raise NotImplementedError()

    def update(self, reward: int, history: List[Action], opp_history: List[Action]):
        pass
