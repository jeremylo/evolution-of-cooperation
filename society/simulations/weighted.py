from random import choices
from typing import Dict, List

import numpy as np

from society.action import Action
from society.agent import Agent
from society.ipd import PAYOFF_MATRIX, Match


class WeightedNetworkSimulation:
    agents: Dict[int, Agent]
    rewards: List[List[List[int]]]
    action_histories: List[List[List[Action]]]
    reward_histories: List[List[int]]

    def __init__(self, agents: List[Agent], weights: np.ndarray) -> None:
        self.agents = agents
        self.weights = weights

        self.population = len(self.agents)
        for i, agent in enumerate(self.agents):
            agent.set_index(i, self.population)

    def reset(self) -> None:
        self.rewards = [
            [[] for j in range(self.population)] for i in range(self.population)
        ]
        self.action_histories = [
            [[] for j in range(self.population)] for i in range(self.population)
        ]
        self.reward_histories = [[] for i in range(self.population)]

    def play_move(self, a, b, train=True):
        move1 = self.agents[a].gameplay_strategy.play_move(
            self.action_histories[a][b], self.action_histories[b][a]
        )
        move2 = self.agents[b].gameplay_strategy.play_move(
            self.action_histories[b][a], self.action_histories[a][b]
        )

        self.action_histories[a][b].append(move1)
        self.action_histories[b][a].append(move2)

        reward1, reward2 = PAYOFF_MATRIX[(move1, move2)]

        self.rewards[a][b].append(reward1)
        self.rewards[b][a].append(reward2)

        self.reward_histories[a].append(reward1)
        self.reward_histories[b].append(reward2)

        if train:
            self.agents[a].gameplay_strategy.update(
                reward1, self.action_histories[a][b], self.action_histories[b][a]
            )
            self.agents[b].gameplay_strategy.update(
                reward2, self.action_histories[b][a], self.action_histories[a][b]
            )

    def play_round(self, train=True) -> None:
        for index in range(self.population):
            partner = choices(
                range(self.population),
                weights=self.weights[index],
                k=1,
            )[0]

            if index != partner:
                self.play_move(index, partner, train)
            else:
                self.rewards[index][index].append(0)
