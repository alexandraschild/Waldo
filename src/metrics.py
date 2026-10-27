import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def AUSC(results_dir, benchmark, model, position=""):
    position_suffix = f"_{position}" if position else ""
    
    # reference file (no added_field)
    ref_path = os.path.join(results_dir, f"{benchmark}_pred_eval_{model}_field0{position_suffix}.json")
    with open(ref_path, "r") as f:
        ref_data = json.load(f)

    # keep only IDs that are correct in the reference file
    correct_ids = [ex["id"] for ex in ref_data if ex["score"] == 1.0]

    added_fields = [0.5, 1, 3, 5]
    accuracies = [(0, 1.0)]

    for af in added_fields:
        path = os.path.join(results_dir, f"{benchmark}_pred_eval_{model}_field{af}{position_suffix}.json")
        if not os.path.exists(path):
            print(f"Warning: missing file {path}")
            continue

        with open(path, "r") as f:
            data = json.load(f)

        acc = [ex["score"] for ex in data if data["id"].isin(correct_ids)].mean()

        accuracies.append((af, acc))

    # sort by added_field
    accuracies.sort(key=lambda x: x[0])

    xs, ys = zip(*accuracies)

    # numerical integration (trapezoidal rule)
    area = np.trapz(ys, xs)

    # maximum possible area = width of x-range * max accuracy (1.0)
    max_area = (max(xs) - min(xs)) * 1.0
    coeff = area / max_area if max_area > 0 else 0

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o", label=f"{model}")
    plt.fill_between(xs, ys, alpha=0.2)
    plt.xlabel("Added Field")
    plt.ylabel("Accuracy")
    plt.title(f"Scaling Curve ({benchmark}, {model})\nAUSC={coeff:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return area, accuracies


def PVS(results_dir, benchmark, model, position=""):
    position_suffix = f"_{position}" if position else ""

    # reference file (no added_field)
    ref_path = os.path.join(results_dir, f"{benchmark}_pred_eval_{model}_field0{position_suffix}.json")
    with open(ref_path, "r") as f:
        ref_data = json.load(f)

    # IDs that were wrong in the reference file
    wrong_ids = {ex["id"] for ex in ref_data if ex["score"] == 0.0}

    added_fields = [0.5, 1, 3, 5]
    recovery_stats = []

    for af in added_fields:
        path = os.path.join(results_dir, f"{benchmark}_pred_eval_{model}_field{af}{position_suffix}.json")
        if not os.path.exists(path):
            print(f"Warning: missing file {path}")
            continue

        with open(path, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df = df[df["id"].isin(wrong_ids)]

        if len(df) == 0:
            recovered = 0
        else:
            recovered = (df["score"] == 1.0).sum() / len(df)

        recovery_stats.append((af, recovered, len(wrong_ids)))

    return recovery_stats

