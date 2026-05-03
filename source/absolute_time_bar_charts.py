import numpy as np
import matplotlib.pyplot as plt
import os

base_folders = {
    "Vectorised Pairwise": "source/Results/vectorised",
    "FMM": "source/Results/fmm",
    r"BH, $\theta=0$": "source/Results/bh_0",
    r"BH, $\theta=0.5$": "source/Results/bh_0.5",
    r"BH, $\theta=1$": "source/Results/bh_1",
}

virial = "1"

targets = [10, 1000]  # N values we want


def load_time(folder, target_N):
    for file in os.listdir(folder):
        parts = file.replace(".npy", "").split("_")
        N = float(parts[1])

        if N == target_N:
            data = np.load(os.path.join(folder, file))
            return data[0], data[1]  # mean, std




# store results
results = {N: {"means": [], "stds": []} for N in targets}
labels = list(base_folders.keys())

for label, base in base_folders.items():
    folder = f"{base}/{virial}"

    for N in targets:
        mean, std = load_time(folder, N)
        results[N]["means"].append(mean)
        results[N]["stds"].append(std)


# plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

for ax, N in zip(axes, targets):

    means = results[N]["means"]
    stds = results[N]["stds"]
    x = np.arange(len(labels))

    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')

    ax.set_title(f"N = {N}")
    ax.set_ylabel("Time (s)")

plt.tight_layout()
plt.savefig("plots/bar_comparison.png", dpi=600)
plt.show()