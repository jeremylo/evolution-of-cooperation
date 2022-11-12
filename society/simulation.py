from typing import Dict, List

from society.agent import Agent
from society.ipd import Match


class Simulation:
    agents: Dict[int, Agent]
    returns: Dict[int, Dict[int, float]]

    def __init__(self, agents: List[Agent]) -> None:
        self.agents = agents
        self.returns = [[[0] for _ in self.agents] for _ in self.agents]

        population = len(self.agents)
        for i, agent in enumerate(self.agents):
            agent.set_index(i, population)

    def play_round(self) -> None:
        for index, agent in enumerate(self.agents):
            partner_index = agent.select_partner(self.returns[index])
            if index != partner_index:
                score1, score2 = Match(
                    agent.gameplay_strategy,
                    self.agents[partner_index].gameplay_strategy,
                ).play(limit=100)

                self.returns[index][partner_index].append(score1)
                self.returns[partner_index][index].append(score2)
