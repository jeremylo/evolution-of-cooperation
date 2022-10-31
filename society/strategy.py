from typing import Dict, List

from society.action import Action


class GameplayStrategy:
    def on_match_start(self):
        pass

    def on_match_end(self):
        pass

    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        raise NotImplementedError()


class SelectionStrategy:
    def select(self, returns: Dict[int, float]) -> int:
        raise NotImplementedError()
