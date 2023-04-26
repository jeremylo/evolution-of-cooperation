import json
import pickle
import sys


def main(outfile, filenames):
    obj = {
        "train_society_rewards": {},
        "train_peaks": {},
        "train_mean_reward": {},
        "test_society_rewards": {},
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

                if key not in obj["train_society_rewards"]:
                    obj["train_society_rewards"][key] = []
                    obj["test_society_rewards"][key] = []

                obj["train_society_rewards"][key].append(
                    result["train_society_rewards"]
                )
                obj["test_society_rewards"][key].append(result["test_society_rewards"])

        with open(outfile, "wb") as f:
            pickle.dump(obj, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
