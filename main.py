from Simulation.simulation import Simulation

simulation = Simulation(100, 1e-5, 1e-7, 1, 1e-2)
while True:
    simulation.make_iteration()
