from enum import IntEnum
from random import choice


class Action(IntEnum):
    COOPERATE = 0
    DEFECT = 1


def flip_action(action: Action) -> Action:
    return Action.COOPERATE if action == Action.DEFECT else Action.DEFECT


def sample_random_action() -> Action:
    return choice([Action.COOPERATE, Action.DEFECT])
