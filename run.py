import os
import sys
import csv
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from utils.data_preprocessing import dose_class, load_data, LABEL_KEY

# Import student submission
import submission


def run(args, data, learner, large_error_penalty=False):
    avg = []
    frac_incorrect = []
    print("Running {}".format(args.model))
    for _ in range(args.runs):
        # Shuffle
        data = data.sample(frac=1)
        T = len(data)
        n_egregious = 0
        correct = np.zeros(T, dtype=bool)
        for t in range(T):
            x = dict(data.iloc[t])
            label = x.pop(LABEL_KEY)
            action = learner.choose(x)
            correct[t] = action == dose_class(label)
            reward = int(correct[t]) - 1
            if (action == "low" and dose_class(label) == "high") or (
                action == "high" and dose_class(label) == "low"
            ):
                n_egregious += 1
                reward = large_error_penalty
            learner.update(x, action, reward)

        results = {
            "total_fraction_correct": np.mean(correct),
            "average_fraction_incorrect": np.mean(
                [np.mean(~correct[:t]) for t in range(1, T)]
            ),
            "fraction_incorrect_per_time": [np.mean(~correct[:t]) for t in range(1, T)],
            "fraction_egregious": float(n_egregious) / T,
        }
        avg.append(results["fraction_incorrect_per_time"])
        print([(x, results[x]) for x in results if x != "fraction_incorrect_per_time"])

    frac_incorrect.append((args.model, np.mean(np.asarray(avg), 0)))
    return frac_incorrect


def plot_frac_incorrect(frac_incorrect):
    plt.xlabel("examples seen")
    plt.ylabel("fraction_incorrect")
    legend = []
    for name, values in frac_incorrect:
        legend.append(name)
        plt.plot(values[10:])
    plt.ylim(0.0, 1.0)
    plt.legend(legend)
    plt.savefig(os.path.join("results", "fraction_incorrect.png"))


def main(args):
    # Import data and define features
    data = load_data()

    features = [
        "Age in decades",
        "Height (cm)",
        "Weight (kg)",
        "Male",
        "Female",
        "Asian",
        "Black",
        "White",
        "Unknown race",
        "Carbamazepine (Tegretol)",
        "Phenytoin (Dilantin)",
        "Rifampin or Rifampicin",
        "Amiodarone (Cordarone)",
    ]

    extra_features = [
        "VKORC1AG",
        "VKORC1AA",
        "VKORC1UN",
        "CYP2C912",
        "CYP2C913",
        "CYP2C922",
        "CYP2C923",
        "CYP2C933",
        "CYP2C9UN",
    ]

    features = features + extra_features

    # Run the appropriate model based on user inputs
    frac_incorrect = []
    if args.model == "fixed":
        frac_incorrect = run(args, data, submission.FixedDosePolicy())

    if args.model == "clinical":
        frac_incorrect = run(args, data, submission.ClinicalDosingPolicy())

    if args.model == "linucb":
        frac_incorrect = run(
            args,
            data,
            submission.LinUCB(3, features, alpha=args.alpha),
            large_error_penalty=args.large_error_penalty,
        )

    if args.model == "egreedy":
        frac_incorrect = run(
            args,
            data,
            submission.eGreedyLinB(3, features, alpha=args.ep),
            large_error_penalty=args.large_error_penalty,
        )

    if args.model == "thompson":
        frac_incorrect = run(
            args,
            data,
            submission.ThomSampB(3, features, alpha=args.v2),
            large_error_penalty=args.large_error_penalty,
        )

    # Store results based on frac_incorrect
    os.makedirs("results", exist_ok=True)
    if frac_incorrect != []:
        for algorithm, results in frac_incorrect:
            with open(f"results/{algorithm}.csv", "w", newline="") as f:
                csv.writer(f).writerows(results.reshape(-1, 1).tolist())

    # Concatenate all model results
    frac_incorrect_all = []
    for filename in os.listdir("results"):
        if filename.endswith(".csv"):
            algorithm = filename.split(".")[0]
            with open(os.path.join("results", filename), "r") as f:
                frac_incorrect_all.append(
                    (
                        algorithm,
                        np.array(list(csv.reader(f))).astype("float64").squeeze(),
                    )
                )

    # Plot the fraction of incorrect results
    plot_frac_incorrect(frac_incorrect_all)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--model", choices=["fixed", "clinical", "linucb", "egreedy", "thompson"]
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ep", type=float, default=1)
    parser.add_argument("--v2", type=float, default=0.001)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--large-error-penalty", type=float, default=-1)
    args = parser.parse_args()
    main(args)
