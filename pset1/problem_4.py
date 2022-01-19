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
    sim = Simulation(N=50, E=0.1, size=1000, radius=20, masses=1, visualise=False)
    sim.run_simulation(run_until_steadystate=True)
    sim.run_simulation(seconds=1000)

    total_pressure = 0
    for i in range(sim.N):
        if len(sim.wall_momenta[i]) > 1 and len(sim.wall_times[i]) > 1:
            average_momentum = np.mean(sim.wall_momenta[i][1:])
            times = np.array(sim.wall_times[i][1:]) - np.array(sim.wall_times[i][:-1])
            average_time = np.mean(times)
            pressure = average_momentum / sim.size
            total_pressure += pressure

        if len(sim.wall_momenta_up[i]) > 1 and len(sim.wall_times_up[i]) > 1:
            average_momentum = np.mean(sim.wall_momenta_up[i][1:])
            times = np.array(sim.wall_times_up[i][1:]) - np.array(sim.wall_times_up[i][:-1])
            average_time = np.mean(times)
            pressure = average_momentum / sim.size
            total_pressure += pressure

        if len(sim.wall_momenta_down[i]) > 1 and len(sim.wall_times_down[i]) > 1:
            average_momentum = np.mean(sim.wall_momenta_down[i][1:])
            times = np.array(sim.wall_times_down[i][1:]) - np.array(sim.wall_times_down[i][:-1])
            average_time = np.mean(times)
            pressure = average_momentum / sim.size
            total_pressure += pressure

        if len(sim.wall_momenta_left[i]) > 1 and len(sim.wall_times_left[i]) > 1:
            average_momentum = np.mean(sim.wall_momenta_left[i][1:])
            times = np.array(sim.wall_times_left[i][1:]) - np.array(sim.wall_times_left[i][:-1])
            average_time = np.mean(times)
            pressure = average_momentum / sim.size
            total_pressure += pressure

    # sim.pressure = []

    # sim.run_simulation(seconds=10000)

    speeds = np.sqrt(np.sum(sim.vel**2, axis=1))
    v_rms = np.sqrt(np.mean(speeds))

    n = sim.N / sim.size**2
    kBT = 0.5 * sim.masses[0] * v_rms**2

    print(n * kBT)
    print(total_pressure / 1000)
    print(total_pressure / (1000 * n * kBT))


if __name__ == "__main__":
    main()
