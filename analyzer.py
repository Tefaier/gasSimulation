import time

import numpy as np
import matplotlib.pyplot as plt

from Simulation.simulation import Simulation


k_B = 1.38e-23

class GasSimulationAnalyzer:
    def __init__(self, simulation, mass=1.0):
        self.simulation = simulation
        self.mass = mass

    def calculate_speeds(self):
        velocities = self.simulation.molecules_vel
        speeds = np.linalg.norm(velocities, axis=1)
        return speeds

    def calculate_temperature(self):
        speeds = self.calculate_speeds()
        avg_kinetic_energy = 0.5 * self.mass * np.mean(speeds ** 2)
        temperature = (2 * avg_kinetic_energy) / (3 * k_B)
        return temperature

    def plot_speed_distribution(self, bins=30):
        speeds = self.calculate_speeds()
        max_speed = np.max(speeds)
        T = self.calculate_temperature()  # Вычисляем температуру из скоростей
        m = self.mass

        # Гистограмма
        plt.hist(speeds, bins=bins, density=True, alpha=0.6, color='g', label='Simulation Data')

        # Теоретическое распределение
        v = np.linspace(0, max_speed * 1.5, 1000)
        a = np.sqrt(m / (2 * np.pi * k_B * T))
        maxwell_dist = 4 * np.pi * (v ** 2) * (a ** 3) * np.exp(- (m * v ** 2) / (2 * k_B * T))

        plt.plot(v, maxwell_dist, 'r-', label='Maxwell Distribution')
        plt.title(f'Speed Distribution (T = {T:.2f} K)')
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()

    @staticmethod
    def calculate_pressure(container_volume, time_step, wall_collision_impulse):
        pressure = wall_collision_impulse / (container_volume * time_step)
        return pressure

    def calculate_theoretic_pressure(self, n, pressure_area):
        T = self.calculate_temperature()
        return n * k_B * T / pressure_area


def calculate_avg_vel_by_temperature(temp, mass):  # Kelvins
    avg_kinetic_energy = 3 * k_B * temp / 2
    avg_speed = np.sqrt(2 * avg_kinetic_energy / mass)
    return avg_speed



if __name__ == "__main__":
    n = 10000
    radius = 1.88e-10
    weight = 6.63e-26
    temperature = 300 # Kelvins
    start_speed = calculate_avg_vel_by_temperature(temperature, weight)
    volume = 1000 * n * (4 / 3) * np.pi * (radius ** 3) # usual gas molecules take up 0,07% of volume
    simulation = Simulation(n, radius, weight, start_speed, volume)
    analyzer = GasSimulationAnalyzer(simulation, mass=weight)

    momentum = 0
    for i in range(80000):
        simulation.make_iteration()
        momentum += simulation.last_iteration_border_momentum_effect
    analyzer.plot_speed_distribution(bins=50)

    cube_area = 6 * volume ** (2/3)
    pressure = analyzer.calculate_pressure(cube_area, simulation.time_since_start, momentum)
    print("Давление эмпирическое:", pressure)
    print("Давление теоретическое:", analyzer.calculate_theoretic_pressure(n, volume))

    print(f"PV = {pressure * volume}")
    print(f"Nk_BT = {n * 1.38e-23 * temperature}")