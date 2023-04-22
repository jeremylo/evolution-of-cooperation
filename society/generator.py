import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
from typing import Callable, List

import numpy as np
from tqdm.auto import tqdm

from society.agent import Agent
from society.simulations.adaptive import AdaptiveSimulation
from society.util import compute_policies, find_weight_peaks, generate_population


def do_run(
    generate_agents: Callable[[int], List[Agent]],
    population: int,
    train_rounds: int = 10_000,
    test_rounds: int = 10_000,
    payoff_matrix=None,
    metadata=None,
    generation_args=None,
) -> dict:
    # Generate a new population
    agents, weights_matrix, G = generate_population(
        generate_agents, population, generation_args
    )

    for agent in agents:
        agent.train()

    # Run a number of rounds
    sim = AdaptiveSimulation(agents, weights_matrix, payoff_matrix)

    # Run the training phase
    train_society_rewards = []

    sim.reset()
    for _ in range(train_rounds):
        r = sim.play_round(train=True)
        train_society_rewards.append(r)

    train_weights = sim.produce_weight_matrix().copy()
    train_cooperation = sim.calculate_cooperativeness_proportions().copy()
    train_returns = [sum(rewards) for rewards in sim.reward_histories]
    train_peaks = find_weight_peaks(train_weights)
    train_mean_reward = np.mean(list(chain(*sim.reward_histories)))

    agent_policies = compute_policies(agents)

    # Run the testing phase
    test_society_rewards = []

    for agent in agents:
        agent.eval()

    sim.reset()
    for _ in range(test_rounds):
        r = sim.play_round(train=False)
        test_society_rewards.append(r)

    test_weights = sim.produce_weight_matrix().copy()
    test_cooperation = sim.calculate_cooperativeness_proportions().copy()
    test_returns = [sum(rewards) for rewards in sim.reward_histories]
    test_peaks = find_weight_peaks(test_weights)
    test_mean_reward = np.mean(list(chain(*sim.reward_histories)))

    return {
        "population": population,
        "agents": agents,
        "agent_args": generation_args,
        "agent_policies": agent_policies,
        "train_rounds": train_rounds,
        "train_society_rewards": train_society_rewards,
        "train_mean_reward": train_mean_reward,
        "train_weights": train_weights,
        "train_cooperation": train_cooperation,
        "train_returns": train_returns,
        "train_peaks": train_peaks,
        "test_rounds": test_rounds,
        "test_society_rewards": test_society_rewards,
        "test_mean_reward": test_mean_reward,
        "test_weights": test_weights,
        "test_cooperation": test_cooperation,
        "test_returns": test_returns,
        "test_peaks": test_peaks,
        "metadata": metadata,
    }


def do_runs(args, max_workers=24):
    with tqdm(total=len(args)) as p:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(do_run, *a) for a in args]

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                train_peaks = (
                    result["train_peaks"][0][result["train_peaks"][1].argsort()[::-1]]
                    if result["train_peaks"][0] is not None
                    else []
                )

                test_peaks = (
                    result["test_peaks"][0][result["test_peaks"][1].argsort()[::-1]]
                    if result["test_peaks"][0] is not None
                    else []
                )

                p.write(
                    f'#{i + 1:<6} {result["population"]:<6} {result["train_mean_reward"]:<10.3f} {result["test_mean_reward"]:<10.3f} {str([round(r, 3) for r in train_peaks]):<28} {str([round(r, 3) for r in test_peaks]):<20} {result["agent_args"]}'
                )

                p.update(1)

                yield result


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)
