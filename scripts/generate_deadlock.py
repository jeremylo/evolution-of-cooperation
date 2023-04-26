import json
from datetime import datetime

from society.action import Action
from society.agents.qlearning import DoubleTabularQLearner
from society.generator import Encoder, do_runs

KWARGS = [
    # {"lookback": 1, "epsilon": 0.1, "learning_rate": 0.05, "discount_factor": 0.99},
    # {"lookback": 2, "epsilon": 0.1, "learning_rate": 0.05, "discount_factor": 0.99},
    {"lookback": 3, "epsilon": 0.1, "learning_rate": 0.05, "discount_factor": 0.99},
]

GAME = "DEADLOCK"
PAYOFF_MATRIX = {
    (Action.COOPERATE, Action.COOPERATE): (1, 1),
    (Action.COOPERATE, Action.DEFECT): (0, 3),
    (Action.DEFECT, Action.COOPERATE): (3, 0),
    (Action.DEFECT, Action.DEFECT): (2, 2),
}


def generate_agents(population, **kwargs):
    return [DoubleTabularQLearner(**kwargs) for _ in range(population)]


def main():
    print("GAME:", GAME)
    print("Agent parameters:", KWARGS)

    args = [
        (generate_agents, size, 10_000, 10_000, PAYOFF_MATRIX, {}, kwargs)
        for kwargs in KWARGS
        for size in (16, 32, 64, 128, 256)
        for _ in range(20)
    ]

    dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    with open(f"results/{dt} - {GAME}.txt", "w") as f:
        for result in do_runs(args, max_workers=30):
            obj = {
                **result,
                "agents": [str(a) for a in result["agents"]],
                "game": GAME,
            }

            f.write(json.dumps(obj, cls=Encoder) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
