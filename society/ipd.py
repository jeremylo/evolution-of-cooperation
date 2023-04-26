from random import random

import numpy as np

from society.action import Action, flip_action
from society.agent import Agent

PAYOFF_MATRIX = {
    (Action.COOPERATE, Action.COOPERATE): (3, 3),
    (Action.COOPERATE, Action.DEFECT): (0, 5),
    (Action.DEFECT, Action.COOPERATE): (5, 0),
    (Action.DEFECT, Action.DEFECT): (1, 1),
}


def mutate_move(action: Action, noise: float):
    if 0 < random() < noise:
        return flip_action(action)

    return action


class Match:
    def __init__(self, agent1: Agent, agent2: Agent) -> None:
        self.agent1 = agent1
        self.agent2 = agent2

        self.total_moves = np.clip(np.random.geometric(0.00346), 1, 1000)

    def play_moves(self, continuation_probability: float, limit: int, noise: float):
        score1 = 0
        score2 = 0

        history1 = []
        history2 = []

        for _ in range(min(self.total_moves, limit)):
            move1 = mutate_move(self.agent1.play_move(history1, history2), noise)
            move2 = mutate_move(self.agent2.play_move(history2, history1), noise)

            increase1, increase2 = PAYOFF_MATRIX[(move1, move2)]
            score1 += increase1
            score2 += increase2

            history1.append(move1)
            history2.append(move2)

            yield (move1, move2), (score1, score2), (increase1, increase2)

    def play(
        self, continuation_probability: float = 1, limit: int = 500, noise: float = 0
    ):
        *_, (moves, scores, rewards) = self.play_moves(
            continuation_probability=continuation_probability, limit=limit, noise=noise
        )
        return scores
