# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt
import numpy as np
from time import time

# create a new class for the simulation with some randomish variable choices
sim = Simulation(N=100, E=100, size=750, radius=3, mass=1, delay=0, visualise=False)

vels = [np.sqrt(sum(p.vel**2)) for p in sim.particles]
# print("INITIAL", vels)
plt.figure()


INITIAL_STEPS = 2500


# run the simulation
start = time()
vels = []
sim.run_simulation(steps=INITIAL_STEPS)
vels.extend([np.sqrt(sum(p.vel**2)) for p in sim.particles])
for _ in range(5):
    sim.run_simulation(steps=INITIAL_STEPS // 10)
    vels.extend([np.sqrt(sum(p.vel**2)) for p in sim.particles])
vels = np.array(vels)
print("Runtime: {:1.2f}s".format(time() - start))

v_range = np.linspace(0, np.ceil(vels.max()), 1000)
v_rms = np.sqrt(np.mean(vels**2))

kBT = v_rms**2 / 3

plt.plot(v_range, v_range / kBT * np.exp(-v_range**2 / (2 * kBT)), label="Analytic distribution")

plt.hist(vels, bins="fd", density=True, label="Simulated distribution")

plt.xlabel(r"Velocity, $v$")
plt.ylabel(r"$\mathrm{d}N/\mathrm{d}v$")

plt.legend()

# print("FINAL", vels)
plt.show()