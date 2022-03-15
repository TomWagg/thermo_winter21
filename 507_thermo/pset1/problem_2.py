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


def velocity_dist(v, v_rms):
    return v / (0.5 * v_rms**2) * np.exp(-(v / v_rms)**2)


def energy_dist(E, m, v_rms):
    kBT = 0.5 * m * v_rms**2
    return 1 / kBT * np.exp(-E / kBT)


def part_b(seconds=10000):
    assert seconds >= 100

    # create a new class for the simulation with some randomish variable choices
    sim = Simulation(N=300, E=1, size=750, radius=3, masses=1, delay=10, visualise=False)

    # start a timer and create an empty velocity array
    start = time()
    velocities = []

    # run the simulation until the given number of steps and save the new velocities
    sim.run_simulation(seconds=seconds)
    velocities.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))

    # keep running the simulation 100 more times but for 100x less time each
    # motivation behind this is to increase the sample size and make a nicer histogram
    for _ in range(100):
        sim.run_simulation(seconds=seconds // 100)
        velocities.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))
    velocities = np.array(velocities)
    print("Runtime: {:1.2f}s".format(time() - start))

    # save the velocities for later (just in case)
    np.save("data/vels_2b.npy", velocities)

    # work out the value of k_B T from velocities
    v_rms = np.sqrt(np.mean(velocities**2))

    # start a figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # plot the analytic velocity distribution
    v_range = np.linspace(0, np.ceil(velocities.max()), 1000)
    axes[0].plot(v_range, velocity_dist(v_range, v_rms), label="Analytic distribution", lw=5)

    # plot a histogram of the velocities
    axes[0].hist(velocities, bins="fd", density=True, label="Simulated distribution",
                 color="tab:purple", alpha=0.8)

    # label axes, add a legend
    axes[0].set_xlabel(r"Speed, $v \, [\rm cm \, s^{-1}]$")
    axes[0].set_ylabel(r"$\mathrm{d}N/\mathrm{d}v$")
    axes[0].legend()

    # repeat for the energies
    energies = 0.5 * velocities**2
    E_range = np.linspace(0, np.ceil(energies.max()), 1000)
    axes[1].plot(E_range, energy_dist(E_range, 1, v_rms), label="Analytic distribution", lw=5)

    axes[1].hist(energies, bins="fd", density=True, label="Simulated distribution",
                 color="tab:purple", alpha=0.8)

    axes[1].set_xlabel(r"Energy, $E \, [\rm erg]$")
    axes[1].set_ylabel(r"$\mathrm{d}N/\mathrm{d}E$")
    axes[1].legend()

    plt.savefig("figures/2b.pdf", format="pdf", bbox_inches="tight")

    plt.show()


def part_c(seconds=10000):
    # create a new class for the simulation with some randomish variable choices
    N = 300
    low_mass = 1
    high_mass = 10
    masses = np.concatenate([np.repeat(low_mass, N // 2), np.repeat(high_mass, N - N // 2)])
    sim = Simulation(N=N, E=1, size=750, radius=3, masses=masses, delay=1, visualise=False)

    # start a timer and create an empty velocity array
    start = time()
    velocities = []

    # run the simulation until the given number of steps and save the new velocities
    sim.run_simulation(seconds=seconds)
    velocities.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))

    # keep running the simulation 100 more times but for 100x less time each
    # motivation behind this is to increase the sample size and make a nicer histogram
    for _ in range(100):
        sim.run_simulation(seconds=seconds // 100)
        velocities.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))
    velocities = np.array(velocities)
    print("Runtime: {:1.2f}s".format(time() - start))

    # save the velocities for later (just in case)
    np.save("data/vels_2c.npy", velocities)

    # -- PLOTTING --
    # tile out the masses to the full range
    full_masses = np.tile(masses, 101)

    # work out the value of k_B T from velocities
    v_rms_low = np.sqrt(np.mean(velocities[full_masses == low_mass]**2))
    v_rms_high = np.sqrt(np.mean(velocities[full_masses == high_mass]**2))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # plot the analytic Maxwellian
    v_range = np.linspace(0, np.ceil(velocities.max()), 1000)
    axes[0].plot(v_range, 0.5 * (velocity_dist(v_range, v_rms_low)
                                 + velocity_dist(v_range, v_rms_high)), label="Mixture of Maxwellians", lw=3)
    axes[0].plot(v_range, 0.5 * velocity_dist(v_range, v_rms_low), label="Individual Maxwellians",
                 linestyle="dotted", color="darkblue", lw=2)
    axes[0].plot(v_range, 0.5 * velocity_dist(v_range, v_rms_high),
                 linestyle="dotted", color="darkblue", lw=2)

    # plot a histogram of the velocities
    axes[0].hist(velocities, bins="fd", density=True, label="Simulated distribution",
                 color="tab:orange", alpha=0.8)

    # label axes, add a legend
    axes[0].set_xlabel(r"Speed, $v \, [\rm cm \, s^{-1}]$")
    axes[0].set_ylabel(r"$\mathrm{d}N/\mathrm{d}v$")
    axes[0].legend()

    # repeat for energies
    energies = 0.5 * full_masses * velocities**2
    E_range = np.linspace(0, np.ceil(energies.max()), 1000)
    analytic = 0.5 * (energy_dist(E_range, low_mass, v_rms_low) + energy_dist(E_range, high_mass, v_rms_high))
    axes[1].plot(E_range, analytic, label="Analytic distribution", lw=3)

    # plot a histogram of the velocities
    axes[1].hist(energies, bins="fd", density=True, label="Simulated distribution",
                 color="tab:orange", alpha=0.8)

    # label axes, add a legend
    axes[1].set_xlabel(r"Energy, $E \, [\rm erg]$")
    axes[1].set_ylabel(r"$\mathrm{d}N/\mathrm{d}E$")
    axes[1].legend()

    plt.savefig("figures/2c.pdf", format="pdf", bbox_inches="tight")

    plt.show()


def main():
    # part_b()
    part_c()


if __name__ == "__main__":
    main()
