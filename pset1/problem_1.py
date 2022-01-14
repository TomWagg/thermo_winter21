# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt
import numpy as np

# create a new class for the simulation with some randomish variable choices
sim = Simulation(N=100, E=1, size=750, radius=3, mass=1, delay=0, visualise=True)

vels = [np.sqrt(sum(p.vel**2)) for p in sim.particles]
print("INITIAL", vels)
plt.figure()


# run the simulation
sim.run_simulation(steps=1000)
vels = [np.sqrt(sum(p.vel**2)) for p in sim.particles]
plt.hist(vels, bins="fd", density=True)
print("FINAL", vels)
plt.show()