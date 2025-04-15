from Simulation.display import show_molecules
from Simulation.simulation import Simulation

n = 2
radius = 1.88e-10
weight = 6.63e-26
max_speed = 398 * 10e-12
volume = 10e-27

simulation = Simulation(n, radius, weight, max_speed, volume)
# simulation.force_border_speed(4, 5e-8)
while True:
    simulation.make_iteration()
    show_molecules(simulation, 10, 10)
