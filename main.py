from Simulation.display import show_molecules
from Simulation.simulation import Simulation
import numpy as np

from analyzer import heat_up_demonstration, calculate_avg_vel_by_temperature, GasSimulationAnalyzer, \
    default_demonstration, move_demonstration

if __name__ == "__main__":
    np.setbufsize(2 ** 16)
    n = 40000
    radius = 1.88e-10
    weight = 6.63e-26
    temperature = 300  # Kelvins
    start_speed = calculate_avg_vel_by_temperature(temperature, weight)
    volume = 1000 * n * (4 / 3) * np.pi * (radius ** 3)  # usual gas molecules take up 0,07% of volume
    simulation = Simulation(n, radius, weight, start_speed, volume)
    analyzer = GasSimulationAnalyzer(simulation, mass=weight)
    move_demonstration(simulation, analyzer)

