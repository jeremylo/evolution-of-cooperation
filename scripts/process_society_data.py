import json
import pickle
import sys

import numpy as np


def main(outfile, filenames):
    obj = {
        "train_weights": {},
        "train_peaks": {},
        "train_mean_reward": {},
        "test_weights": {},
        "test_peaks": {},
        "test_mean_reward": {},
    }

    for filename in filenames:
        with open(filename) as f:
            for line in f:
                result = json.loads(line)

                key = (
                    result["population"],
                    result["agent_args"]["lookback"],
                    result["agent_args"]["epsilon"],
                )

                if key not in obj["train_weights"]:
                    obj["train_weights"][key] = []
                    obj["train_peaks"][key] = []
                    obj["train_mean_reward"][key] = []
                    obj["test_weights"][key] = []
                    obj["test_peaks"][key] = []
                    obj["test_mean_reward"][key] = []

                obj["train_weights"][key].append(result["train_weights"])
                obj["test_weights"][key].append(result["test_weights"])

                train_peaks = (
                    list(
                        np.array(result["train_peaks"][0])[
                            np.array(result["train_peaks"][1]).argsort()[::-1]
                        ]
                    )
                    if result["train_peaks"][0]
                    else [1.0]
                )

                test_peaks = (
                    list(
                        np.array(result["test_peaks"][0])[
                            np.array(result["test_peaks"][1]).argsort()[::-1]
                        ]
                    )
                    if result["test_peaks"][0]
                    else [1.0]
                )

                obj["train_peaks"][key].append(train_peaks)
                obj["test_peaks"][key].append(test_peaks)

                obj["train_mean_reward"][key].append(result["train_mean_reward"])
                obj["test_mean_reward"][key].append(result["test_mean_reward"])

        with open(outfile, "wb") as f:
            pickle.dump(obj, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
