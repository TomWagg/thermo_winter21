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
    sim = Simulation(N=300, E=1, size=750, radius=3, masses=1, delay=10, visualise=False)

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
    np.save("vels.npy", velocities)

    # work out the value of k_B T from velocities
    v_rms = np.sqrt(np.mean(velocities**2))
    kBT = v_rms**2 / 2

    # start a figure
    fig, ax = plt.subplots()

    # plot the analytic Maxwellian
    v_range = np.linspace(0, np.ceil(velocities.max()), 1000)
    ax.plot(v_range, velocity_dist(v_range, kBT), label="Analytic distribution")

    # plot a histogram of the velocities
    ax.hist(velocities, bins="fd", density=True, label="Simulated distribution")

    # label axes, add a legend
    ax.set_xlabel(r"Velocity, $v$")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}v$")
    ax.legend()

    plt.show()

def part_c(steps=10000):

    # create a new class for the simulation with some randomish variable choices
    N = 300
    sim = Simulation(N=N, E=0.1, size=750, radius=3,
                     masses=np.concatenate([np.repeat(1, N // 2), np.repeat(100, N - N // 2)]),
                     delay=1, visualise=True)

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
    np.save("vels_2c.npy", velocities)

    # work out the value of k_B T from velocities
    v_rms = np.sqrt(np.mean(velocities**2))
    kBT = v_rms**2 / 2

    # start a figure
    fig, ax = plt.subplots()

    # plot the analytic Maxwellian
    v_range = np.linspace(0, np.ceil(velocities.max()), 1000)
    ax.plot(v_range, velocity_dist(v_range, kBT), label="Analytic distribution")

    # plot a histogram of the velocities
    ax.hist(velocities, bins="fd", density=True, label="Simulated distribution")

    # label axes, add a legend
    ax.set_xlabel(r"Velocity, $v$")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}v$")
    ax.legend()

    plt.show()

def main():
    part_c(10000)


if __name__ == "__main__":
    main()
