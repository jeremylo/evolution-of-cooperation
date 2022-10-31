from random import random

from society.action import Action, flip_action
from society.strategy import GameplayStrategy

PAYOFF_MATRIX = {
    (Action.COOPERATE, Action.COOPERATE): (3, 3),
    (Action.COOPERATE, Action.DEFECT): (0, 5),
    (Action.DEFECT, Action.COOPERATE): (5, 0),
    (Action.DEFECT, Action.DEFECT): (1, 1),
}


class Match:
    def __init__(
        self, strategy1: GameplayStrategy, strategy2: GameplayStrategy
    ) -> None:
        self.strategy1 = strategy1
        self.strategy2 = strategy2

    def _mutate(self, action: Action, noise: float):
        if 0 < random() < noise:
            return flip_action(action)

        return action

    def play_moves(self, continuation_probability: float, limit: int, noise: float):
        score1 = 0
        score2 = 0

        history1 = []
        history2 = []

        self.strategy1.on_match_start()
        self.strategy2.on_match_start()

        i = 0
        while i < limit and (i < 1 or random() < continuation_probability):
            move1 = self._mutate(self.strategy1.play_move(history1, history2), noise)
            move2 = self._mutate(self.strategy2.play_move(history2, history1), noise)

            increase1, increase2 = PAYOFF_MATRIX[(move1, move2)]
            score1 += increase1
            score2 += increase2

            history1.append(move1)
            history2.append(move2)

            i += 1
            yield (move1, move2), (score1, score2), (increase1, increase2)

        self.strategy1.on_match_end()
        self.strategy2.on_match_end()

    def play(
        self, continuation_probability: float = 1, limit: int = 500, noise: float = 0
    ):
        *_, (moves, scores, rewards) = self.play_moves(
            continuation_probability=continuation_probability, limit=limit, noise=noise
        )
        return scores
