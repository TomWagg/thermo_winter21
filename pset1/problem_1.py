# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt
import numpy as np
from time import time


def main():

    # create a new class for the simulation with some randomish variable choices
    sim = Simulation(N=200, E=1, size=750, radius=3, masses=1, delay=0, visualise=False)
    # sim.run_simulation()
    # return
    vels = np.sqrt(np.sum(sim.vel**2, axis=1))

    print("initial v_rms", np.sqrt(np.mean(vels**2)))

    plt.figure()

    INITIAL_STEPS = 10000

    # run the simulation
    start = time()
    vels = []
    sim.run_simulation(steps=INITIAL_STEPS)
    print(sim.get_total_kinetic_energy())
    vels.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))
    for _ in range(100):
        sim.run_simulation(steps=INITIAL_STEPS // 100)
        vels.extend(np.sqrt(np.sum(sim.vel**2, axis=1)))
    vels = np.array(vels)
    print("Runtime: {:1.2f}s".format(time() - start))

    v_range = np.linspace(0, np.ceil(vels.max()), 1000)
    v_rms = np.sqrt(np.mean(vels**2))

    print("final v_rms", np.sqrt(np.mean(vels**2)))

    kBT = v_rms**2 / 2

    plt.plot(v_range, v_range / kBT * np.exp(-v_range**2 / (2 * kBT)), label="Analytic distribution")

    plt.hist(vels, bins="fd", density=True, label="Simulated distribution")

    plt.xlabel(r"Velocity, $v$")
    plt.ylabel(r"$\mathrm{d}N/\mathrm{d}v$")

    plt.legend()

    plt.show()

    np.save("vels_alt.npy", vels)


if __name__ == "__main__":
    main()