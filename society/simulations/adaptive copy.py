from random import choices
from typing import Dict, List

import numpy as np

from society.action import Action
from society.agent import Agent
from society.ipd import PAYOFF_MATRIX

K = 50


class AdaptiveSimulation:
    agents: Dict[int, Agent]
    rewards: List[List[List[int]]]
    action_histories: List[List[List[Action]]]
    reward_histories: List[List[int]]

    def __init__(self, agents: List[Agent], weights: np.ndarray) -> None:
        self.agents = agents
        self.weights = weights

        self.population = len(self.agents)

        self.connections = [
            [j for j in range(self.population) if self.weights[i, j] > 0]
            for i in range(self.population)
        ]

    def reset(self) -> None:
        self.rewards = [
            [[] for j in range(self.population)] for i in range(self.population)
        ]
        self.action_histories = [
            [[] for j in range(self.population)] for i in range(self.population)
        ]
        self.reward_histories = [[] for i in range(self.population)]
        self.mutual_cooperations = np.zeros((self.population, self.population))

    def play_move(self, a, b, train=True):
        move1 = self.agents[a].play_move(
            self.action_histories[a][b], self.action_histories[b][a]
        )
        move2 = self.agents[b].play_move(
            self.action_histories[b][a], self.action_histories[a][b]
        )

        self.action_histories[a][b].append(move1)
        self.action_histories[b][a].append(move2)

        reward1, reward2 = PAYOFF_MATRIX[(move1, move2)]

        self.rewards[a][b].append(reward1)
        self.rewards[b][a].append(reward2)

        self.reward_histories[a].append(reward1)
        self.reward_histories[b].append(reward2)

        if move1 == move2 == Action.COOPERATE:
            self.mutual_cooperations[a, b] += 1
            self.mutual_cooperations[b, a] += 1

        if train:
            self.agents[a].update(
                reward1, self.action_histories[a][b], self.action_histories[b][a]
            )
            self.agents[b].update(
                reward2, self.action_histories[b][a], self.action_histories[a][b]
            )

    def calculate_weights(self):
        return [
            [
                (
                    1 - np.mean(self.action_histories[index][p][-K:])
                    if self.rewards[index][p]
                    else 0
                )
                + 0.1
                for p in self.connections[index]
            ]
            for index in range(self.population)
        ]

    def play_round(self, train=True) -> None:
        weights = self.calculate_weights()

        for index in range(self.population):
            partner = choices(
                self.connections[index],
                weights=weights[index],
                k=1,
            )[0]

            if index != partner:
                self.play_move(index, partner, train)
            else:
                self.rewards[index][index].append(0)
