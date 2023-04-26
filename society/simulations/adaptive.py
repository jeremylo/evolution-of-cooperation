from random import choices, sample
from typing import Dict, List

import numpy as np

from society.action import Action
from society.agent import Agent
from society.ipd import PAYOFF_MATRIX


class AdaptiveSimulation:
    agents: Dict[int, Agent]
    rewards: List[List[List[int]]]
    action_histories: List[List[List[Action]]]
    reward_histories: List[List[int]]

    def __init__(
        self,
        agents: List[Agent],
        weights: np.ndarray,
        payoff_matrix=None,
        threshold: float = 0.05,
    ) -> None:
        self.agents = agents
        self.population = len(self.agents)

        self.connections = [
            [j for j in range(self.population) if weights[i, j] > 0]
            for i in range(self.population)
        ]

        self.weights = None
        self.differences = []

        self.payoff_matrix = (
            payoff_matrix if payoff_matrix is not None else PAYOFF_MATRIX
        )
        self.threshold = threshold

    def reset(self) -> None:
        """Resets the simulation."""

        self.rewards = [
            [[] for j in range(self.population)] for i in range(self.population)
        ]
        self.action_histories = [
            [[] for j in range(self.population)] for i in range(self.population)
        ]
        self.reward_histories = [[] for i in range(self.population)]
        self.mutual_cooperations = np.zeros((self.population, self.population))
        self.frequencies = np.zeros((self.population, self.population))

    def play_move(self, a, b, train=True):
        """Simulates an interaction between two agents.

        Args:
            a (int): An agent index.
            b (int): The opponent's index.
            train (bool, optional): Whether to update agent weights. Defaults to True.

        Returns:
            int: The combined reward.
        """

        move1 = self.agents[a].play_move(
            self.action_histories[a][b], self.action_histories[b][a]
        )
        move2 = self.agents[b].play_move(
            self.action_histories[b][a], self.action_histories[a][b]
        )

        self.action_histories[a][b].append(move1)
        self.action_histories[b][a].append(move2)

        reward1, reward2 = self.payoff_matrix[(move1, move2)]

        self.rewards[a][b].append(reward1)
        self.rewards[b][a].append(reward2)

        self.reward_histories[a].append(reward1)
        self.reward_histories[b].append(reward2)

        if move1 == Action.COOPERATE and move2 == Action.COOPERATE:
            self.mutual_cooperations[a, b] += 1
            self.mutual_cooperations[b, a] += 1

        if train:
            self.agents[a].update(
                reward1, self.action_histories[a][b], self.action_histories[b][a]
            )
            self.agents[b].update(
                reward2, self.action_histories[b][a], self.action_histories[a][b]
            )

        return reward1 + reward2

    def calculate_cooperativeness_proportions(self):
        """Calculates the cooperativeness matrix.

        Returns:
            np.ndarray: A matrix.
        """

        return np.divide(
            self.mutual_cooperations,
            self.frequencies,
            out=np.zeros_like(self.mutual_cooperations),
            where=self.frequencies != 0,
        )

    def calculate_weights(self):
        """Computes the weight matrix.

        Returns:
            np.ndarray: The weight matrix.
        """

        return np.maximum(
            np.divide(
                self.mutual_cooperations,
                self.frequencies,
                out=np.zeros_like(self.mutual_cooperations),
                where=self.frequencies != 0,
            ),
            self.threshold,
        )

    def get_weights_by_agent(self, weights=None):
        """Produces a list of weights for the connections of each agent.

        Args:
            weights (np.ndarray, optional): A precomputed weight matrix. Defaults to None.

        Returns:
            List[List[float]]: A list of agents' edge weight lists.
        """

        if weights is None:
            weights = self.calculate_weights()

        return [weights[i, self.connections[i]] for i in range(self.population)]

    def produce_weight_matrix(self):
        """Produces the weight matrix.

        Returns:
            np.ndarray: The society weight matrix.
        """

        weights = self.get_weights_by_agent()

        matrix = np.full((self.population, self.population), -np.inf)
        for index in range(self.population):
            for partner, weight in zip(self.connections[index], weights[index]):
                if index != partner:
                    matrix[index, partner] = weight

        return matrix

    def play_round(self, train=True) -> None:
        """Simulates a single round.

        Args:
            train (bool, optional): Whether to update agent weights. Defaults to True.

        Returns:
            int: The aggregated society reward.
        """

        weights = self.calculate_weights()

        if self.weights is not None:
            self.differences.append(np.sum(weights - self.weights))

        self.weights = weights

        r = 0

        for index in sample(range(self.population), k=self.population):
            partner = choices(
                self.connections[index],
                weights=weights[index, self.connections[index]],
                k=1,
            )[0]

            self.frequencies[index, partner] += 1
            self.frequencies[partner, index] += 1

            if index != partner:
                r += self.play_move(index, partner, train)
            else:
                self.rewards[index][index].append(0)

        return r
