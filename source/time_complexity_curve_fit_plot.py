import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit


base_folders = {
    "Vectorised Pairwise": "source/Results/vectorised",
    "FMM": "source/Results/fmm",
    r"BH, $\theta = 0$": "source/Results/bh_0",
    r"BH, $\theta = 0.5$": "source/Results/bh_0.5",
    r"BH, $\theta = 1$": "source/Results/bh_1",
}

virial = "1"



def nlogn(x, A):
    return A * x * np.log(x)

def powerlaw(x, A, b):
    return A * x**b


def r2(y, yfit):
    ss_res = np.sum((y - yfit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot



for label, base in base_folders.items():

    folder = f"{base}/{virial}"

    Ns, means, stds = [], [], []

    for file in os.listdir(folder):
        
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
    Ns, means, stds = Ns[idx], means[idx], stds[idx]


    xdata = Ns
    ydata = means

    # n log n fit
    popt1, _ = curve_fit(nlogn, xdata, ydata, p0=[1e-6])
    yfit1 = nlogn(xdata, *popt1)
    r2_1 = r2(ydata, yfit1)

    # power law fit
    popt2, _ = curve_fit(powerlaw, xdata, ydata, p0=[1e-6, 2])
    yfit2 = powerlaw(xdata, *popt2)
    r2_2 = r2(ydata, yfit2)


    # smooth curves
    xfit = np.linspace(min(xdata), max(xdata), 300)



    plt.figure(figsize=(6, 4))

    plt.errorbar(Ns, means, yerr=stds, fmt='x', markersize=4)

    plt.plot(
        xfit,
        nlogn(xfit, *popt1),
        linewidth = 0.7,
        label=fr"$A N\log N$: A={popt1[0]:.2e}, $R^2$={r2_1:.3f}"
    )

    plt.plot(
        xfit,
        powerlaw(xfit, *popt2),
        linewidth = 0.7,
        label=fr"$A N^b$: A={popt2[0]:.2e}, b={popt2[1]:.2f}, $R^2$={r2_2:.3f}"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("N")
    plt.ylabel("Time (s)")
    plt.title(f"{label}")

    plt.legend(fontsize=8)
    plt.tight_layout()

    filename = f"plots/virial1_{label.replace(' ', '_').replace('$', '').replace('\\', '')}.png"
    plt.savefig(filename, dpi=600)
    plt.show()