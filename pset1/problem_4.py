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
    mass = 1
    sim = Simulation(N=100, E=1, size=1000, radius=20, masses=mass, visualise=False)
    sim.run_simulation(run_until_steadystate=True)

    sim.wall_momenta = []

    seconds = 1000
    sim.run_simulation(seconds=seconds)

    total_pressure = np.array(sim.wall_momenta).sum() / sim.size / seconds

    speeds = np.sqrt(np.sum(sim.vel**2, axis=1))
    v_rms = np.sqrt(np.mean(speeds))

    n = sim.N / sim.size**2
    kBT = 0.5 * mass * v_rms**2

    print(n * kBT)
    print(total_pressure)
    print(total_pressure / (n * kBT))


if __name__ == "__main__":
    main()
