# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt
import numpy as np
from time import time


def t_relax_analytic(N, size, radius, E, mass):
    n = N / size**2
    sigma = radius * 2
    v0 = np.sqrt(2 * E / mass)
    return (n * sigma * v0)**(-1)


def plot_N_comparison(size=1000, radius=20, E=0.1, mass=1):
    start = time()

    fig, ax = plt.subplots()

    N_range = np.array([25, 50, 75, 100, 125, 150, 175, 200])

    for N in N_range:
        t_relax = []
        for _ in range(5):
            sim = Simulation(N=N, E=E, size=size, radius=radius, masses=mass, visualise=False)
            t_relax.append(sim.run_simulation(run_until_steadstate=True))

        ax.errorbar(N, np.median(t_relax), xerr=0.0, yerr=[[np.median(t_relax) - np.min(t_relax)],
                                                           [np.max(t_relax) - np.median(t_relax)]],
                    color="tab:orange")
        ax.scatter(N, np.median(t_relax), color="tab:orange")
        print("N =", N, "done", t_relax)

    N_range_smooth = np.linspace(N_range.min(), N_range.max(), 1000)
    ax.plot(N_range_smooth, t_relax_analytic(N_range_smooth, size, radius, E, mass))
    ax.set_xlabel(r"Number of Particles")
    ax.set_ylabel(r"Relaxation Time, $\tau_{\rm relax} \, [\rm s]$")

    print("Runtime {:1.2f}s".format(time() - start))

    plt.savefig("figures/3c_compare_N.png", bbox_inches="tight")

    plt.show()


def plot_r_comparison(N=100, size=1000, E=0.1, mass=1):
    start = time()

    fig, ax = plt.subplots()

    r_range = np.array([1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])

    for radius in r_range:
        t_relax = []
        for _ in range(5):
            sim = Simulation(N=N, E=E, size=size, radius=radius, masses=mass, visualise=False)
            t_relax.append(sim.run_simulation(run_until_steadstate=True))

        ax.errorbar(radius, np.median(t_relax), xerr=0.0, yerr=[[np.median(t_relax) - np.min(t_relax)],
                                                                [np.max(t_relax) - np.median(t_relax)]],
                    color="tab:orange")
        ax.scatter(radius, np.median(t_relax), color="tab:orange")
        print("r =", radius, "done", t_relax)

    r_range_smooth = np.linspace(r_range.min(), r_range.max(), 1000)
    ax.plot(r_range_smooth, t_relax_analytic(r_range_smooth, size, radius, E, mass))
    ax.set_xlabel(r"Radius, $r \, [\rm cm]$")
    ax.set_ylabel(r"Relaxation Time, $\tau_{\rm relax} \, [\rm s]$")

    print("Runtime {:1.2f}s".format(time() - start))

    plt.savefig("figures/3c_compare_radius.png", bbox_inches="tight")

    plt.show()


def main():
    plot_N_comparison()
    plot_r_comparison()


if __name__ == "__main__":
    main()
