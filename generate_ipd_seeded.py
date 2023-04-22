import json
from datetime import datetime

from society.agents.constant import AllC, AllD
from society.agents.pavlov import Pavlov
from society.agents.qlearning import DoubleTabularQLearner
from society.agents.random import Random
from society.agents.tft import TitForTat
from society.generator import Encoder, do_runs

KWARGS = {"lookback": 2, "epsilon": 0.1, "learning_rate": 0.05, "discount_factor": 0.99}

GAME = "IPD-SEEDED"
PAYOFF_MATRIX = None


def generate_agents_tft(population, seed_proportion, **kwargs):
    n = int(seed_proportion * population)
    return [TitForTat() for _ in range(n)] + [
        DoubleTabularQLearner(**kwargs) for _ in range(population - n)
    ]


def generate_agents_allc(population, seed_proportion, **kwargs):
    n = int(seed_proportion * population)
    return [AllC() for _ in range(n)] + [
        DoubleTabularQLearner(**kwargs) for _ in range(population - n)
    ]


def generate_agents_alld(population, seed_proportion, **kwargs):
    n = int(seed_proportion * population)
    return [AllD() for _ in range(n)] + [
        DoubleTabularQLearner(**kwargs) for _ in range(population - n)
    ]


def generate_agents_random(population, seed_proportion, **kwargs):
    n = int(seed_proportion * population)
    return [Random() for _ in range(n)] + [
        DoubleTabularQLearner(**kwargs) for _ in range(population - n)
    ]


def generate_agents_pavlov(population, seed_proportion, **kwargs):
    n = int(seed_proportion * population)
    return [Pavlov() for _ in range(n)] + [
        DoubleTabularQLearner(**kwargs) for _ in range(population - n)
    ]


GENERATORS = {
    "TitForTat": generate_agents_tft,
    "AllC": generate_agents_allc,
    "AllD": generate_agents_alld,
    "Random": generate_agents_random,
    "Pavlov": generate_agents_pavlov,
}


def main():
    print("GAME:", GAME)
    print("Agent parameters:", KWARGS)

    args = [
        (
            generate_agents,
            128,
            10_000,
            10_000,
            PAYOFF_MATRIX,
            {"seeded_by": seeder},
            {**KWARGS, "seed_proportion": proportion},
        )
        for seeder, generate_agents in GENERATORS.items()
        for proportion in (0.05, 0.1, 0.25, 0.5)
        for _ in range(20)
    ]

    dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    with open(f"results/{dt} - {GAME}.txt", "w") as f:
        for result in do_runs(args, max_workers=10):
            obj = {
                **result,
                "agents": [str(a) for a in result["agents"]],
                "game": GAME,
            }

            f.write(json.dumps(obj, cls=Encoder) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
