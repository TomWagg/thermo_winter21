# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from time import time

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 18

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)


def t_relax_analytic(N, size, radius, E, mass):
    n = N / size**2
    sigma = radius * 2
    v0 = np.sqrt(2 * E / mass)
    return (n * sigma * v0)**(-1)


def plot_N_comparison(fig, ax, size=1000, radius=20, E=0.1, mass=1, repeats=10):
    start = time()

    N_range = np.array([25, 50, 75, 100, 125, 150, 175, 200])

    t_relax_list = []
    for N in N_range:
        t_relax = []
        for _ in range(repeats):
            sim = Simulation(N=N, E=E, size=size, radius=radius, masses=mass, visualise=False)
            t_relax.append(sim.run_simulation(run_until_steadstate=True))

        ax.errorbar(N, np.median(t_relax), xerr=0.0, yerr=[[np.median(t_relax) - np.min(t_relax)],
                                                           [np.max(t_relax) - np.median(t_relax)]],
                    color="tab:orange")
        ax.scatter(N, np.median(t_relax), color="tab:orange")
        print("N =", N, "done", t_relax)
        t_relax_list.append(t_relax)

    np.save("data/t_relax_N.npy", t_relax_list)

    N_range_smooth = np.linspace(N_range.min(), N_range.max(), 1000)
    ax.plot(N_range_smooth, t_relax_analytic(N_range_smooth, size, radius, E, mass))
    ax.set_xlabel(r"Number of Particles")
    ax.set_ylabel(r"Relaxation Time, $\tau_{\rm relax} \, [\rm s]$")

    print("Runtime {:1.2f}s".format(time() - start))

    return fig, ax


def plot_r_comparison(fig, ax, N=100, size=1000, E=0.1, mass=1, repeats=10):
    start = time()

    r_range = np.array([10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30])

    t_relax_list = []
    for radius in r_range:
        t_relax = []
        for _ in range(repeats):
            sim = Simulation(N=N, E=E, size=size, radius=radius, masses=mass, visualise=False)
            t_relax.append(sim.run_simulation(run_until_steadstate=True))

        ax.errorbar(radius, np.median(t_relax), xerr=0.0, yerr=[[np.median(t_relax) - np.min(t_relax)],
                                                                [np.max(t_relax) - np.median(t_relax)]],
                    color="tab:orange")
        ax.scatter(radius, np.median(t_relax), color="tab:orange")
        print("r =", radius, "done", t_relax)
        t_relax_list.append(t_relax)

    np.save("data/t_relax_r.npy", t_relax_list)

    r_range_smooth = np.linspace(r_range.min(), r_range.max(), 1000)
    ax.plot(r_range_smooth, t_relax_analytic(N=N, size=size, radius=r_range_smooth, E=E, mass=mass))
    ax.set_xlabel(r"Radius, $r \, [\rm cm]$")
    ax.set_ylabel(r"Relaxation Time, $\tau_{\rm relax} \, [\rm s]$")

    print("Runtime {:1.2f}s".format(time() - start))

    return fig, ax


def main():

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    fig, axes[0] = plot_N_comparison(fig=fig, ax=axes[0])
    fig, axes[1] = plot_r_comparison(fig=fig, ax=axes[1])

    plt.savefig("figures/3c.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
