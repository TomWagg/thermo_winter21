# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt
import numpy as np
from time import time


def velocity_dist(v, kBT):
    return v / kBT * np.exp(-v**2 / (2 * kBT))


def part_b(steps=10000):
    assert steps >= 100

    # create a new class for the simulation with some randomish variable choices
    sim = Simulation(N=300, E=1, size=750, radius=3, masses=1, delay=10, visualise=True)

    # start a timer and create an empty velocity array
    start = time()
    velocities = []

    # run the simulation until the given number of steps and save the new velocities
    sim.run_simulation(steps=steps)
    velocities.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))

    # keep running the simulation 100 more times but for 100x less time each
    # motivation behind this is to increase the sample size and make a nicer histogram
    for _ in range(100):
        sim.run_simulation(steps=steps // 100)
        velocities.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))
    velocities = np.array(velocities)
    print("Runtime: {:1.2f}s".format(time() - start))

    # save the velocities for later (just in case)
    np.save("data/vels_2b.npy", velocities)

    # work out the value of k_B T from velocities
    v_rms = np.sqrt(np.mean(velocities**2))
    kBT = v_rms**2 / 2

    # start a figure
    fig, ax = plt.subplots()

    # plot the analytic Maxwellian
    v_range = np.linspace(0, np.ceil(velocities.max()), 1000)
    ax.plot(v_range, velocity_dist(v_range, kBT), label="Analytic distribution", lw=5)

    # plot a histogram of the velocities
    ax.hist(velocities, bins="fd", density=True, label="Simulated distribution",
            color="tab:purple", alpha=0.8)

    # label axes, add a legend
    ax.set_xlabel(r"Velocity, $v$")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}v$")
    ax.legend()

    plt.show()


def part_c(steps=10000):
    # create a new class for the simulation with some randomish variable choices
    N = 300
    masses = np.concatenate([np.repeat(1, N // 2), np.repeat(10, N - N // 2)])
    sim = Simulation(N=N, E=1, size=750, radius=3, masses=masses, delay=1, visualise=False)

    # start a timer and create an empty velocity array
    start = time()
    velocities = []

    # run the simulation until the given number of steps and save the new velocities
    sim.run_simulation(steps=steps)
    velocities.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))

    # keep running the simulation 100 more times but for 100x less time each
    # motivation behind this is to increase the sample size and make a nicer histogram
    for _ in range(100):
        sim.run_simulation(steps=steps // 100)
        velocities.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))
    velocities = np.array(velocities)
    print("Runtime: {:1.2f}s".format(time() - start))

    # save the velocities for later (just in case)
    np.save("data/vels_2c.npy", velocities)

    # -- PLOTTING --
    # tile out the masses to the full range
    full_masses = np.tile(masses, 101)

    # work out the value of k_B T from velocities
    kBT_low = 0.5 * np.mean(velocities[full_masses == 1]**2)
    kBT_high = 0.5 * np.mean(velocities[full_masses == 10]**2)

    # start a figure
    fig, ax = plt.subplots()

    # plot the analytic Maxwellian
    v_range = np.linspace(0, np.ceil(velocities.max()), 1000)
    ax.plot(v_range, 0.5 * (velocity_dist(v_range, kBT_low)
                            + velocity_dist(v_range, kBT_high)), label="Mixture of Maxwellians")

    # plot a histogram of the velocities
    ax.hist(velocities, bins="fd", density=True, label="Simulated distribution")

    # label axes, add a legend
    ax.set_xlabel(r"Velocity, $v \, [\rm cm \, s^{-1}]$")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}v$")
    ax.legend()

    plt.savefig("figures/2c_velocity.png", bbox_inches="tight")
    plt.show()

    # start a figure
    fig, ax = plt.subplots()

    energies = 0.5 * full_masses * velocities**2

    # plot the analytic Maxwellian
    # v_range = np.linspace(0, np.ceil(velocities.max()), 1000)
    # ax.plot(v_range, 0.5 * (velocity_dist(v_range, kBT_low)
    #                         + velocity_dist(v_range, kBT_high)), label="Mixture of Maxwellians")

    # plot a histogram of the velocities
    ax.hist(energies, bins="fd", density=True, label="Simulated distribution")

    # label axes, add a legend
    ax.set_xlabel(r"Energy, $E \, [\rm erg]$")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}E$")
    ax.legend()

    plt.savefig("figures/2c_energy.png", bbox_inches="tight")
    plt.show()


def main():
    part_c(1000)


if __name__ == "__main__":
    main()
