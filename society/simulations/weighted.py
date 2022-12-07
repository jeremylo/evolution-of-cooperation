from random import choices
from typing import Dict, List

import networkx as nx
import numpy as np

from society.agent import Agent
from society.ipd import Match


class WeightedNetworkSimulation:
    agents: Dict[int, Agent]
    returns: Dict[int, Dict[int, float]]

    def __init__(self, agents: List[Agent], weights: np.ndarray) -> None:
        self.agents = agents
        self.weights = weights

        self.population = len(self.agents)
        for i, agent in enumerate(self.agents):
            agent.set_index(i, self.population)

    def reset(self) -> None:
        self.returns = [
            [[] for j in range(self.population)] for i in range(self.population)
        ]

    def play_round(self, limit=1) -> None:
        for index, agent in enumerate(self.agents):
            partner = choices(range(self.population), weights=self.weights[index], k=1)[0]

            if index != partner:
                score1, score2 = Match(
                    agent.gameplay_strategy,
                    self.agents[partner].gameplay_strategy,
                ).play(limit=limit)

                self.returns[index][partner].append(score1)
                self.returns[partner][index].append(score2)
            else:
                self.returns[index][index].append(0)
