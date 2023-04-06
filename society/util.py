from datetime import datetime
from itertools import product
from typing import Callable, List

import networkx as nx
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from society.action import Action
from society.agent import Agent


def get_timestamp() -> str:
    return datetime.isoformat(datetime.now()).replace(":", "-").replace("T", " ")


def compute_policies(agents: List[Agent]):
    policies = [
        agent._q_table.argmax(axis=-1) if hasattr(agent, "_q_table") else None
        for agent in agents
    ]

    return [
        "".join(
            [
                ("C", "D")[policy[p]]
                for p in product(
                    *[range(4) for _ in range(len(agent._q_table.shape) - 1)]
                )
            ]
        )
        if policy is not None
        else agent.__class__.__name__
        for policy, agent in zip(policies, agents)
    ]


def calculate_cooperativeness(history):
    count = history.count(Action.COOPERATE)

    return count / len(history)


def generate_population(generate_agents: Callable[[int], List[Agent]], population: int):
    agents = generate_agents(population)

    G = nx.complete_graph(population)

    weights_matrix = np.zeros((population, population))
    for u, v, d in G.edges(data=True):
        weights_matrix[u, v] = weights_matrix[v, u] = 1
        try:
            d["weight"] = weights_matrix[u, v]
        except:
            d["weight"] = 0

    return agents, weights_matrix, G


def find_weight_peaks(weights):
    r = np.arange(-0.1, 1.1, 0.001)

    try:
        kernel = gaussian_kde([w for w in weights.ravel() if w >= 0])
        kde = kernel(r)
        peaks, peak_properties = find_peaks(kde, height=0.25, prominence=(0.03, None))

        return r[peaks], kde[peaks]
    except:
        return None, None
