from Simulation.display import show_molecules
from Simulation.simulation import Simulation
import matplotlib.pyplot as plt

simulation = Simulation(2, 0.2, 1e-7, 1, 1)
while True:
    simulation.make_iteration()
    # show_molecules(simulation, 10, 10)
