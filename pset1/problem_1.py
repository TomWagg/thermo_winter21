# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt
import numpy as np
from time import time


def main():

    # create a new class for the simulation with some randomish variable choices
    sim = Simulation(N=500, E=100, size=750, radius=3, masses=1, delay=0, visualise=False)
    # sim.run_simulation()
    # return
    vels = np.sqrt(np.sum(sim.vel**2, axis=1))

    print("initial v_rms", np.sqrt(np.mean(vels**2)))

    plt.figure()


    INITIAL_STEPS = 1000

    # run the simulation
    start = time()
    vels = []
    sim.run_simulation(steps=INITIAL_STEPS)
    vels.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))
    for _ in range(5):
        sim.run_simulation(steps=INITIAL_STEPS // 10)
        vels.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))
    vels = np.array(vels)
    print("Runtime: {:1.2f}s".format(time() - start))

    v_range = np.linspace(0, np.ceil(vels.max()), 1000)
    v_rms = np.sqrt(np.mean(vels**2))

    print("final v_rms", np.sqrt(np.mean(vels**2)))

    kBT = v_rms**2 / 3

    plt.plot(v_range, v_range / kBT * np.exp(-v_range**2 / (2 * kBT)), label="Analytic distribution")

    plt.hist(vels, bins="fd", density=True, label="Simulated distribution")

    plt.xlabel(r"Velocity, $v$")
    plt.ylabel(r"$\mathrm{d}N/\mathrm{d}v$")

    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()