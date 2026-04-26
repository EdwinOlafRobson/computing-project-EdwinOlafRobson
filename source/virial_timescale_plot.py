import numpy as np
import matplotlib.pyplot as plt

import acceleration_calculation.accelerations as accelerations
import system
import time_step
import physical_tests
import animation


import os
import time
import numpy as np



base_folders = {
    "Vectorised Pairwise": "source/Results/vectorised",
    "FMM": "source/Results/fmm",
    r"BH, $\theta = 0$": "source/Results/bh_0",
    r"BH, $\theta = 0.5$": "source/Results/bh_0.5",
    r"BH, $\theta = 1$": "source/Results/bh_1",
}

virials = ["0.001", "1", "1000"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for ax, virial in zip(axes, virials):

    all_Ns = []

    for label, base in base_folders.items():
        folder = f"{base}/{virial}"

        Ns = []
        means = []
        stds = []

        for file in os.listdir(folder):
            if not file.endswith(".npy"):
                continue

            parts = file.replace(".npy", "").split("_")
            N = float(parts[1])

            data = np.load(os.path.join(folder, file))
            mean, std = data[0], data[1]

            Ns.append(N)
            means.append(mean)
            stds.append(std)

        Ns = np.array(Ns)
        means = np.array(means)
        stds = np.array(stds)

        idx = np.argsort(Ns)
        Ns = Ns[idx]
        means = means[idx]
        stds = stds[idx]

        means_norm = means / means[0]
        stds_norm = stds / means[0]

        all_Ns.extend(Ns)

        ax.errorbar(
            Ns,
            means_norm,
            yerr=stds_norm,
            marker='x',
            markeredgewidth=0.5,
            markersize=4,
            linestyle='none',
            label=label
        )

   
    all_Ns = np.array(all_Ns)
    N_min, N_max = np.min(all_Ns), np.max(all_Ns)

    N_fit = np.logspace(np.log10(N_min), np.log10(N_max), 100)

    fit_linear = (N_fit / N_min)
    fit_quadratic = (N_fit / N_min)**2

    ax.plot(N_fit, fit_linear, linestyle='--', linewidth=1, label=r"$\propto N^1$")
    ax.plot(N_fit, fit_quadratic, linestyle='--', linewidth=1, label=r"$\propto N^2$")

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_title(f"Virial = {virial}")




axes[0].set_xlim(left=9)
axes[0].set_ylim(bottom=0.8)

plt.subplots_adjust(wspace=0)

fig.supxlabel("N")
fig.supylabel(r"Normalised time ($t/t_{0}$)", x=0.08)
fig.suptitle("Runtime Scaling Across Virialisation Regimes")

axes[2].legend(loc = "upper left")

plt.show()