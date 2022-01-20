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


def main():

    an_val = []
    sim_val = []

    N_range = [10, 20, 50, 70, 100]
    for N in N_range:

        mass = 1
        sim = Simulation(N=N, E=0.1, size=1000, radius=20, masses=mass, visualise=False)
        sim.run_simulation(run_until_steadystate=True)

        sim.wall_momenta = []

        seconds = 10000
        sim.run_simulation(seconds=seconds)

        sim_val.append(np.sum(sim.wall_momenta) / (4 * sim.size) / seconds)

        speeds = np.sqrt(np.sum(sim.vel**2, axis=1))
        v_rms = np.sqrt(np.mean(speeds))

        n = sim.N / sim.size**2
        kBT = 0.5 * mass * v_rms**2

        an_val.append(n * kBT)

        print("N =", N, "done")
        print("  ", len(sim.wall_momenta))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].scatter(N_range, an_val, label=r"$n k_B T$")
    axes[0].scatter(N_range, sim_val, label=r"$P$")

    axes[1].scatter(N_range, np.divide(sim_val, an_val), label=r"$P / n k_B T$")

    for ax in axes:
        ax.legend()
        ax.set_xlabel("Number of particles")

    plt.show()


if __name__ == "__main__":
    main()
