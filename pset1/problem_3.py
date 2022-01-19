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


def get_N_comparison(size=1000, radius=20, E=0.1, mass=1, repeats=10):
    start = time()

    N_range = np.array([25, 50, 75, 100, 125, 150, 175, 200])

    t_relax_list = []
    for N in N_range:
        t_relax = []
        for _ in range(repeats):
            sim = Simulation(N=N, E=E, size=size, radius=radius, masses=mass, visualise=False)
            t_relax.append(sim.run_simulation(run_until_steadstate=True))
        print("N =", N, "done", t_relax)
        t_relax_list.append(t_relax)
    np.save("data/t_relax_N.npy", t_relax_list)
    print("Runtime {:1.2f}s".format(time() - start))

    return N_range, t_relax_list


def get_r_comparison(N=100, size=1000, E=0.1, mass=1, repeats=10):
    start = time()

    r_range = np.array([10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30])

    t_relax_list = []
    for radius in r_range:
        t_relax = []
        for _ in range(repeats):
            sim = Simulation(N=N, E=E, size=size, radius=radius, masses=mass, visualise=False)
            t_relax.append(sim.run_simulation(run_until_steadstate=True))
        print("r =", radius, "done", t_relax)
        t_relax_list.append(t_relax)
    np.save("data/t_relax_r.npy", t_relax_list)
    print("Runtime {:1.2f}s".format(time() - start))

    return r_range, t_relax_list


def main():
    N = 100
    radius = 20
    size = 1000
    E = 0.1
    mass = 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    N_range, t_relax_N = get_N_comparison(radius=radius, size=size, E=E, mass=mass)
    r_range, t_relax_r = get_r_comparison(N=N, size=size, E=E, mass=mass)

    ax = axes[0]
    t_relax = t_relax_N

    N_range_smooth = np.linspace(N_range.min(), N_range.max(), 1000)
    ax.plot(N_range_smooth, t_relax_analytic(N_range_smooth, size, radius, E, mass) * 1.2,
            linestyle="dotted", lw=3, color=plt.get_cmap("plasma")(0.8),
            label=r"Analytic function, $X = 1.2$")
    ax.set_xlabel(r"Number of Particles")
    ax.set_ylabel(r"Relaxation Time, $\tau_{\rm relax} \, [\rm s]$")

    ax.errorbar(N_range, np.median(t_relax, axis=1), xerr=0.0,
                yerr=[np.median(t_relax, axis=1) - np.min(t_relax, axis=1),
                      np.max(t_relax, axis=1) - np.median(t_relax, axis=1)],
                marker="o", color=plt.get_cmap("plasma")(0.2), label="Simulated results")

    ax.legend()

    ax = axes[1]
    t_relax = t_relax_r

    r_range_smooth = np.linspace(r_range.min(), r_range.max(), 1000)
    ax.plot(r_range_smooth, t_relax_analytic(N=N, size=size, radius=r_range_smooth, E=E, mass=mass) * 1.2,
            linestyle="dotted", lw=3, color=plt.get_cmap("plasma")(0.8),
            label=r"Analytic function, $X = 1.2$")
    ax.set_xlabel(r"Radius, $r \, [\rm cm]$")
    ax.set_ylabel(r"Relaxation Time, $\tau_{\rm relax} \, [\rm s]$")

    ax.errorbar(r_range, np.median(t_relax, axis=1), xerr=0.0,
                yerr=[np.median(t_relax, axis=1) - np.min(t_relax, axis=1),
                      np.max(t_relax, axis=1) - np.median(t_relax, axis=1)],
                marker="o", color=plt.get_cmap("plasma")(0.2), label="Simulated results")

    ax.legend()

    plt.savefig("figures/3c.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
