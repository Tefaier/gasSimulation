from Simulation.display import show_molecules
from Simulation.simulation import Simulation

n = 2
radius = 1.88e-10
weight = 6.63e-26
max_speed = 398 * 10e-12
volume = 10e-27
# usual gas molecules take up 0,07% of volume
simulation = Simulation(n, radius, weight, max_speed, volume)
while True:
    simulation.make_iteration()
    show_molecules(simulation, 10, 10)
