import numpy as np
import matplotlib.pyplot as plt

virials = ["0.001", "1", "1000"]

labels = [
    r"BH $\theta=0$",
    r"BH $\theta=0.5$",
    r"BH $\theta=1$",
    "Vectorised",
    "FMM"
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

linestyles = ['-', '-', '-', '--', '--']

for axis, virial in zip(axes, virials):

    energies = np.load(f"source/Results/total energy/{virial}/energy.npy")

    for i in range(len(energies)):
        axis.plot(energies[i], label=labels[i], linestyle=linestyles[i], alpha = 0.9 )

    axis.set_title(rf"$r_{{Vir}}$= {virial}")

fig.supxlabel(r"Time ($10^{-2}$s)")
fig.supylabel("Total Energy (J)", x=0.01)

axes[2].legend(loc = "center right")
fig.suptitle("Total Energy In Different Virial Regimes")
plt.tight_layout()
plt.savefig("plots/energy_conservation.png", dpi = 800)
plt.show()