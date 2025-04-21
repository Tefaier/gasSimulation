from Simulation.display import show_molecules
from Simulation.simulation import Simulation
import numpy as np

n = 10000
radius = 1.88e-10
weight = 6.63e-26
max_speed = 398 * 10e-12
volume = 100 * n * (4 / 3) * np.pi * (radius**3)
# usual gas molecules take up 0,07% of volume
simulation = Simulation(n, radius, weight, max_speed, volume)
while True:
    simulation.make_iteration()
    # show_molecules(simulation, 10, 10)
