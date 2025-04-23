import time
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

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
    def calculate_pressure(container_area, time_step, wall_collision_impulse):
        return wall_collision_impulse / (container_area * time_step)

    def calculate_theoretic_pressure(self, n, pressure_area):
        return n * k_B * self.calculate_temperature() / pressure_area

    def calculate_total_energy(self):
        return 0.5 * self.mass * np.sum(self.calculate_speeds() ** 2)


def calculate_avg_vel_by_temperature(temp, mass):  # Kelvins
    avg_kinetic_energy = 3 * k_B * temp / 2
    avg_speed = np.sqrt(2 * avg_kinetic_energy / mass)
    return avg_speed

# border heat up and show of total_energy, temperature, speed distribution, pressure
def heat_up_demonstration(simulation: Simulation, analyzer: GasSimulationAnalyzer):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ims = []
    V = []
    P = []
    T = []
    E = []
    X = []
    iterations = 80000
    update_at_iter = 1000
    momentum = 0
    last_display_time = 0
    bar = tqdm.tqdm(total=iterations)
    for i in range(iterations):
        simulation.make_iteration()
        momentum += simulation.last_iteration_border_momentum_effect
        if i % update_at_iter == 0:
            P.append(analyzer.calculate_pressure(simulation.get_current_area(), simulation.time_since_start - last_display_time, momentum))
            V.append(simulation.get_current_volume())
            T.append(analyzer.calculate_temperature())
            E.append(analyzer.calculate_total_energy())
            X.append(simulation.time_since_start)

            speeds = analyzer.calculate_speeds()

            # Гистограмма
            ax1.hist(speeds, bins=50, density=True, alpha=0.6, color='g', label='Simulation Data')

            # Теоретическое распределение
            v = np.linspace(0, np.max(speeds) * 1.1, 1000)
            a = np.sqrt(analyzer.mass / (2 * np.pi * k_B * T[-1]))
            maxwell_dist = 4 * np.pi * (v ** 2) * (a ** 3) * np.exp(- (analyzer.mass * v ** 2) / (2 * k_B * T[-1]))
            ax1.plot(v, maxwell_dist, 'r-', label='Maxwell Distribution')
            ax1.subtitle(f'Speed Distribution (T = {T[-1]:.2f} K)')
            ax1.xlabel('Speed (m/s)')
            ax1.ylabel('Probability Density')
            ax1.legend()

            ax2.plot(X, T)
            ax2.subtitle('Temperature')
            ax2.xlabel('Time, s')
            ax2.ylabel('T, K')

            momentum = 0
            last_display_time = simulation.time_since_start
        bar.update()
    bar.close()
    ani = anim.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save("heat_up.mp4")

    fig = plt.figure(1, figsize=(9, 6))
    plt.plot(V, P)
    fig.savefig("VP.png")
    fig = plt.figure(1, figsize=(9, 6))
    plt.plot(P, T)
    fig.savefig("PT.png")
    fig = plt.figure(1, figsize=(9, 6))
    plt.plot(V, T)
    fig.savefig("VT.png")



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
    iterations = 80000
    bar = tqdm.tqdm(total=iterations)
    for i in range(iterations):
        simulation.make_iteration()
        momentum += simulation.last_iteration_border_momentum_effect
        bar.update()
    bar.close()
    analyzer.plot_speed_distribution(bins=50)

    cube_area = 6 * volume ** (2/3)
    pressure = analyzer.calculate_pressure(cube_area, simulation.time_since_start, momentum)
    print("Давление эмпирическое:", pressure)
    print("Давление теоретическое:", analyzer.calculate_theoretic_pressure(n, volume))

    print(f"PV = {pressure * volume}")
    print(f"Nk_BT = {n * 1.38e-23 * temperature}")