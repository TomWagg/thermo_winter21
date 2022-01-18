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


def main():

    N = 100
    size = 750
    radius = 15
    E = 0.1
    mass = 1

    start = time()

    fig, ax = plt.subplots()

    N_range = np.array([50, 75, 100, 125, 150])

    for N in N_range:
        t_relax = []
        for _ in range(3):
            sim = Simulation(N=N, E=E, size=size, radius=radius, masses=mass, visualise=False)
            t_relax.append(sim.run_simulation(run_until_steadstate=True))

        ax.errorbar(N, np.median(t_relax), xerr=0.0, yerr=[[np.median(t_relax) - np.min(t_relax)], [np.max(t_relax) - np.median(t_relax)]], color="tab:orange")
        ax.scatter(N, np.median(t_relax), color="tab:orange")
        print("N =", N, "done", t_relax)

    N_range_smooth = np.linspace(N_range.min(), N_range.max(), 1000)
    ax.plot(N_range_smooth, t_relax_analytic(N_range_smooth, size, radius, E, mass))

    print("Runtime {:1.2f}s".format(time() - start))

    plt.show()

    # fig, ax = plt.subplots()

    # r_range = np.array([10, 12, 15, 17, 20, 22, 25])

    # for radius in r_range:
    #     t_relax = []
    #     for _ in range(25):
    #         sim = Simulation(N=N, E=E, size=size, radius=radius, masses=mass, visualise=False)
    #         t_relax.append(sim.run_simulation(run_until_steadstate=True))

    #     ax.errorbar(radius, np.median(t_relax), xerr=0.0, yerr=[[np.median(t_relax) - np.min(t_relax)], [np.max(t_relax) - np.median(t_relax)]], color="tab:orange")
    #     ax.scatter(radius, np.median(t_relax), color="tab:orange")
    #     print("r =", radius, "done", t_relax)

    # r_range_smooth = np.linspace(r_range.min(), r_range.max(), 1000)
    # ax.plot(r_range_smooth, t_relax_analytic(N, size, r_range_smooth, E, mass))

    # print("Runtime {:1.2f}s".format(time() - start))

    # plt.show()


if __name__ == "__main__":
    main()
