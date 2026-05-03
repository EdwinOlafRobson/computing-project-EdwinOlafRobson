import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit


base_folder = "source/Results"

bh_folders = {
    r"$\theta = 0$": f"{base_folder}/bh_0/1",
    r"$\theta = 0.5$": f"{base_folder}/bh_0.5/1",
    r"$\theta = 1$": f"{base_folder}/bh_1/1",
}


# ---------------- models ----------------
def nlogn(x, A):
    return A * x * np.log(x)

def powerlaw(x, A, b):
    return A * x**b


def r2(y, yfit):
    ss_res = np.sum((y - yfit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot


# ---------------- load all data first (for global y-range) ----------------
data_store = {}
all_means = []

for label, folder in bh_folders.items():

    Ns, means, stds = [], [], []

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
    Ns, means, stds = Ns[idx], means[idx], stds[idx]

    data_store[label] = (Ns, means, stds)

    all_means.extend(means)


# global y-limits (critical for fair comparison)
y_min, y_max = np.min(all_means), np.max(all_means)


# ---------------- plotting ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

for ax, (label, (Ns, means, stds)) in zip(axes, data_store.items()):

    xdata = Ns
    ydata = means

    # ---- fits ----
    popt1, _ = curve_fit(nlogn, xdata, ydata, p0=[1e-6])
    yfit1 = nlogn(xdata, *popt1)
    r2_1 = r2(ydata, yfit1)

    popt2, _ = curve_fit(powerlaw, xdata, ydata, p0=[1e-6, 2])
    yfit2 = powerlaw(xdata, *popt2)
    r2_2 = r2(ydata, yfit2)

    xfit = np.linspace(min(xdata), max(xdata), 300)

    # ---- data ----
    ax.errorbar(Ns, means, yerr=stds, fmt='x', markersize=4)

    # ---- fits ----
    ax.plot(
        xfit,
        nlogn(xfit, *popt1),
        linewidth = 0.5,
        label=fr"$A n\log n$ (A={popt1[0]:.2e}, $R^2$={r2_1:.3f})"
    )

    ax.plot(
        xfit,
        powerlaw(xfit, *popt2),
        linewidth = 0.5,
        label=fr"$A n^b$ (A={popt2[0]:.2e}, b={popt2[1]:.2f}, $R^2$={r2_2:.3f})"
    )

    ax.set_title(label)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(9, 1200)
    ax.set_ylim(y_min/2.5, y_max*1.5)
    ax.set_xlabel("N")

    ax.legend(fontsize=7, loc="upper left")


axes[0].set_ylabel("Runtime (s)")


plt.suptitle("Barnes–Hut Scaling")
plt.tight_layout()
plt.savefig("plots/bh_theta_fits.png", dpi=600)
plt.show()