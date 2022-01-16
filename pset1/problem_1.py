# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt
import numpy as np
from time import time


sim = Simulation(N=100, E=1, size=750, radius=15, masses=1, delay=20, visualise=True)

sim.run_simulation()