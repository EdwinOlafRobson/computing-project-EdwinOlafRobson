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

    momenta = np.load(f"source/Results/total momentum/{virial}/momentum.npy")

    for i in range(len(momenta)):
        axis.plot(
            momenta[i],
            label=labels[i],
            linestyle=linestyles[i],
            alpha=0.9
        )

    axis.set_title(fr"$r_{{Vir}}$= {virial}")

fig.supxlabel(r"Time ($10^{-2}$)", y=0.05)
fig.supylabel("Total Normalised Momentum (kgm/s)", x=0.01)

axes[2].legend(loc = "upper right")

fig.suptitle("Normalised Momentum In Different Virial Regiemes")
plt.tight_layout()
plt.savefig("plots/momentum_conservation.png",dpi = 800)
plt.show()