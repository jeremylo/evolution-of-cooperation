from random import random, randrange
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from society.strategy import SelectionStrategy


class QNetwork(nn.Module):
    def __init__(self, population):
        super().__init__()

        self.layer1 = nn.Linear(population, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, population)

        nn.init.kaiming_uniform_(self.layer1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer2.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        x = x.unsqueeze(dim=0)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))

        return x


class QLearningSelectionStrategy(SelectionStrategy):
    _epsilon: float = 0.0

    def __init__(self, population_size: int) -> None:
        self.population_size = population_size
        self._q_network = QNetwork(self.population_size)
        self._learning_rate = 0.001
        self._discount_rate = 0.99

        self._criterion = torch.nn.HuberLoss()
        self._optimiser = optim.Adam(
            self._q_network.parameters(), lr=self._learning_rate  # , weight_decay=1e-5
        )

        self._loss = 0.0
        self._count = 0

    def select_partner(self, returns: List[List[float]]) -> int:
        if random() < self._epsilon:
            return randrange(self.population_size)

        return int(torch.argmax(self._q_network(to_state(returns))))

    def update(
        self, old_state: List[List[float]], new_state: List[List[float]], reward: float
    ):
        target = (
            reward + self._discount_rate * self._q_network(to_state(new_state)).max()
        )

        self._optimiser.zero_grad()
        loss = self._criterion(self._q_network(to_state(old_state)).max(), target)
        self._loss += float(loss)
        loss.backward()
        self._optimiser.step()

        self._count += 1


def to_state(returns):
    return torch.tensor([np.nan_to_num(np.mean(r)) for r in returns], dtype=torch.float)
