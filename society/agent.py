from typing import List

from society.action import Action


class Agent:
    training: bool = True

    def play_move(self, history: List[Action], opp_history: List[Action]) -> Action:
        raise NotImplementedError

    def update(self, reward: int, history: List[Action], opp_history: List[Action]):
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @property
    def parameters(self) -> dict:
        return {}

    def __repr__(self) -> str:
        params = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])

        return f"{self.__class__.__name__}({params})"
