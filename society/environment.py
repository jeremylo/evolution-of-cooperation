from typing import Dict, List

import numpy as np

from society.agent import Agent, TrainableAgent
from society.ipd import Match


class SelectionEnvironment:
    agents: Dict[int, Agent]
    returns: Dict[int, float]

    def __init__(self, agent: TrainableAgent, opponents: List[Agent]) -> None:
        self.agent = agent
        self.agents = [agent] + opponents
        self.population = len(self.agents)

    def reset(self):
        self.returns = []
        for i, agent in enumerate(self.agents):
            agent.set_index(i, self.population)
            self.returns.append([0.0])

        return [np.nan_to_num(np.mean(r, dtype=object)) for r in self.returns]

    def step(self, action: int):
        if action != 0:
            score1, score2 = Match(
                self.agent.gameplay_strategy,
                self.agents[action].gameplay_strategy,
            ).play(limit=100)

            self.returns[action].append(score1)
            return (
                [np.nan_to_num(np.mean(r, dtype=object)) for r in self.returns],
                score1,
                False,
            )

        return [np.nan_to_num(np.mean(r, dtype=object)) for r in self.returns], 0, False
