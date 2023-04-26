import json
import pickle
import sys


def main(outfile, filenames):
    obj = {
        "agent_policies": {},
        "train_returns": {},
        "test_returns": {},
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

                if key not in obj["train_returns"]:
                    obj["agent_policies"][key] = []
                    obj["train_returns"][key] = []
                    obj["test_returns"][key] = []

                obj["agent_policies"][key].append(result["agent_policies"])
                obj["train_returns"][key].append(result["train_returns"])
                obj["test_returns"][key].append(result["test_returns"])

        with open(outfile, "wb") as f:
            pickle.dump(obj, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
