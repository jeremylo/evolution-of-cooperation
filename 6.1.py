import json
from datetime import datetime

import numpy as np

from society.agents.qlearning import DoubleTabularQLearner
from society.generator import do_runs

# KWARGS = {"lookback": 1, "epsilon": 0.1, "learning_rate": 0.05, "discount_factor": 0.99}
# KWARGS = {"lookback": 2, "epsilon": 0.1, "learning_rate": 0.05, "discount_factor": 0.99}
KWARGS = {"lookback": 3, "epsilon": 0.1, "learning_rate": 0.05, "discount_factor": 0.99}
# KWARGS = {"lookback": 4, "epsilon": 0.1, "learning_rate": 0.05, "discount_factor": 0.99}
# KWARGS = {"lookback": 5, "epsilon": 0.1, "learning_rate": 0.05, "discount_factor": 0.99}


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def generate_agents(population):
    return [DoubleTabularQLearner(**KWARGS) for _ in range(population)]


def main():
    print("Agent parameters:", KWARGS)

    args = [
        (generate_agents, size, 10_000, 1_000)
        for size in (512, 1024)  # (16, 32, 64, 128, 256)
        for _ in range(20)
    ]

    dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    with open(f"results/{dt}.txt", "w") as f:
        for result in do_runs(args, max_workers=10):
            obj = {
                **result,
                "agent_args": KWARGS,
                "agents": [str(a) for a in result["agents"]],
            }

            f.write(json.dumps(obj, cls=Encoder) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
