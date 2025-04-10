import matplotlib.pyplot as plt
import numpy as np

from Simulation.simulation import Simulation


def show_molecules(simulation: Simulation, resolution_w: int, resolution_h: int):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    u = np.linspace(0, 2 * np.pi, resolution_w)
    v = np.linspace(0, np.pi, resolution_h)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    positions = simulation.get_current_positions()
    for index, pos in enumerate(positions):
        radius = simulation.molecules_radius[index]
        ax.plot_surface(x * radius + pos[0], y * radius + pos[1], z * radius + pos[2], alpha=0.2)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], marker='^', s=100)
    positions += simulation.molecules_vel * 0.1
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], marker='^', s=100)

    ax.set_xlim(simulation.borders_pos[0], simulation.borders_pos[1])
    ax.set_ylim(simulation.borders_pos[2], simulation.borders_pos[3])
    ax.set_zlim(simulation.borders_pos[4], simulation.borders_pos[5])
    ax.set_aspect('equal')
    plt.show()
