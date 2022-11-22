from typing import Dict, List

import numpy as np

from society.agent import Agent, TrainableAgent
from society.ipd import Match


class OnlineLearningSimulation:
    agents: Dict[int, Agent]
    returns: Dict[int, Dict[int, float]]

    def __init__(self, agents: List[Agent]) -> None:
        self.agents = agents

        self.population = len(self.agents)
        for i, agent in enumerate(self.agents):
            agent.set_index(i, self.population)

    def reset(self) -> None:
        self.returns = [
            [[] for j in range(self.population)] for i in range(self.population)
        ]

    def play_round(self) -> None:
        for index, agent in enumerate(self.agents):
            partner = agent.select_partner(self.returns[index])
            if index != partner:
                old_state = None
                if isinstance(agent, TrainableAgent):
                    old_state = self.returns[index][:]

                score1, score2 = Match(
                    agent.gameplay_strategy,
                    self.agents[partner].gameplay_strategy,
                ).play(limit=100)

                self.returns[index][partner].append(score1)
                self.returns[partner][index].append(score2)

                if isinstance(agent, TrainableAgent):
                    agent.update_selector(old_state, self.returns[index][:], score1)
            else:
                self.returns[index][index].append(0)
