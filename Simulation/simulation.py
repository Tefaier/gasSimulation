import math
from typing import Tuple, Any, Literal

import numpy as np
from queue import PriorityQueue
from tqdm import tqdm
import warnings
from numba import njit

from Simulation.models import Axis
from Simulation.utils import polar_to_cartesian, elastic_balls_interaction, one_sided_elastic_collision, \
    positive_index_to_negative, boltzmann_constant

warnings.filterwarnings('ignore', category=RuntimeWarning)

class Simulation:
    time_since_start: float
    initial_volume: float
    random_initial_pos: bool

    molecules_count: int
    molecules_radius: np.ndarray[Any, np.dtype[np.float64]]
    molecules_weight: np.ndarray[Any, np.dtype[np.float64]]
    molecules_pos: np.ndarray[Any, np.dtype[np.float64]]
    molecules_vel: np.ndarray[Any, np.dtype[np.float64]]
    molecules_pos_time: np.ndarray[Any, np.dtype[np.float64]]
    molecules_history_id: list[int]
    molecules_queue_presence: list[int]
    molecules_closest_interaction_time: list[float | None]

    borders_count: int
    borders_normal: list[Axis] = [Axis.x, Axis.x, Axis.y, Axis.y, Axis.z, Axis.z]
    borders_pos: np.ndarray[Any, np.dtype[np.float64]]
    borders_vel: np.ndarray[Any, np.dtype[np.float64]]
    borders_pos_time: np.ndarray[Any, np.dtype[np.float64]]
    borders_history_id: list[int]
    borders_temperature: list[float | None]

    last_iteration_border_momentum_effect: float

    #  object inside is as follows:
    #  time of interaction
    #  id of first interacting entity
    #  history_id of first entity when interaction was added to queue
    #  id of second interacting entity
    #  history_id of second entity when interaction was added to queue
    interaction_queue: PriorityQueue[Tuple[float, int, int, int, int]]



    def __init__(self, n: int, molecule_radius: float, molecules_weight: float, initial_speed: float, initial_volume: float):
        self.random_initial_pos = False
        self.molecules_count = n
        self.time_since_start = 0
        self.initial_volume = initial_volume
        side_length = initial_volume ** (1/3)
        self.init_molecules(molecule_radius, molecules_weight, side_length * 0.5 - molecule_radius * 1.1, initial_speed)
        self.init_borders(side_length / 2)
        self.interaction_queue = PriorityQueue()
        print("Start of calculating initial interactions")
        bar = tqdm(total = n, ncols=100)
        for molecule_id in range(0, n):
            self.calculate_molecule_interaction(molecule_id)
            bar.update()
        bar.close()

    @staticmethod
    @njit(nopython=True, cache=True)
    def validate_positions(pos_all, rad_all, count) -> bool:
        for molecule_id in range(0, count - 1):
            diff = np.sum((pos_all[molecule_id+1:] - pos_all[molecule_id][np.newaxis, :]) ** 2, axis=1) - (rad_all[molecule_id+1:] + rad_all[molecule_id]) ** 2
            if np.min(diff) < 0:
                return False
        return True

    def init_molecules(self, radius: float, weight: float, max_offset: float, start_speed: float):
        self.molecules_radius = np.ones(shape=(self.molecules_count,)) * radius
        self.molecules_weight = np.ones(shape=(self.molecules_count,)) * weight
        total_volume = np.sum(np.power(self.molecules_radius, 3) * np.pi * 4 / 3)
        print(f"Initiated volume {self.initial_volume}, molecules total volume {total_volume} = {100 * total_volume / self.initial_volume}%")
        counter = 0
        if self.random_initial_pos:
            while True:
                self.molecules_pos = np.random.rand(self.molecules_count, 3) * 2 * max_offset - max_offset
                counter += 1
                if Simulation.validate_positions(self.molecules_pos, self.molecules_radius, self.molecules_count):
                    break
                print(f"Attempt {counter}: failed to generate positions of molecules")
        else:
            per_axis = math.ceil(self.molecules_count ** (1/3))
            xv, yv, zv = np.meshgrid(
                np.linspace(-max_offset, max_offset, per_axis),
                np.linspace(-max_offset, max_offset, per_axis),
                np.linspace(-max_offset, max_offset, per_axis)
            )
            self.molecules_pos = np.concatenate([xv.reshape(per_axis, per_axis, per_axis, 1), yv.reshape(per_axis, per_axis, per_axis, 1), zv.reshape(per_axis, per_axis, per_axis, 1)], axis=3).reshape((per_axis * per_axis * per_axis, 3))[:self.molecules_count]
            if not Simulation.validate_positions(self.molecules_pos, self.molecules_radius, self.molecules_count):
                raise RuntimeError("Failed to use order generation for positions")
        print(f"Attempt {counter}: positions of molecules successfully generated")
        self.molecules_vel = polar_to_cartesian(
            np.ones(shape=(self.molecules_count,)) * start_speed,
            np.random.rand(self.molecules_count) * 2 * np.pi,
            np.random.rand(self.molecules_count) * np.pi
        )
        self.molecules_pos_time = np.zeros(shape=(self.molecules_count,))
        self.molecules_history_id = [0] * self.molecules_count
        self.molecules_queue_presence = [0] * self.molecules_count
        self.molecules_closest_interaction_time = [0] * self.molecules_count

    def init_borders(self, offset: float):
        self.borders_count = len(self.borders_normal)

        self.borders_pos = np.array([-offset, offset] * 3)
        self.borders_vel = np.zeros((self.borders_count,))
        self.borders_pos_time = np.zeros((self.borders_count,))
        self.borders_history_id = [0] * self.borders_count
        self.borders_temperature = [None] * self.borders_count

    def calculate_border_collision_time(self, molecule_id: int, border_id: int) -> float | None:
        axis_mask = self.borders_normal[border_id].value.id
        speed_diff = self.borders_vel[border_id] - self.molecules_vel[molecule_id][axis_mask]
        # if abs(speed_diff) < 1e-8: return None
        return (
                (self.molecules_radius[molecule_id] * (-1 if border_id % 2 == 0 else 1)
                   + self.molecules_pos[molecule_id][axis_mask]
                   - self.borders_pos[border_id]
                   - self.molecules_vel[molecule_id][axis_mask] * self.molecules_pos_time[molecule_id]
                   + self.borders_vel[border_id] * self.borders_pos_time[border_id])
                / speed_diff
        )

    @staticmethod
    @njit(nopython=True, cache=True)
    def calculate_molecule_interaction_with_m(pos_all, vel_all, rad_all, time_all, molecule_id, cutoff, count, to_ignore):
        m_cutoff = cutoff * 1.00000001
        pos = pos_all[molecule_id]
        vel = vel_all[molecule_id]
        rad = rad_all[molecule_id]
        time = time_all[molecule_id]

        # a * t**2 + b * t + c = 0
        vel_diff = vel[np.newaxis, :] - vel_all
        pos_diff = pos_all - pos[np.newaxis, :] + time * vel[np.newaxis, :] - time_all[:, np.newaxis] * vel_all
        a = np.sum(vel_diff ** 2, axis=1)
        b = np.sum(-2 * vel_diff * pos_diff, axis=1)
        c = np.sum(pos_diff ** 2, axis=1) - (rad + rad_all) ** 2
        d = b ** 2 - 4 * a * c
        valid = d > 0
        sqrt_d = np.sqrt(np.where(valid, d, 0))
        solution_1 = np.full((count,), float('inf'))
        solution_2 = np.full((count,), float('inf'))
        solution_1[valid] = (-b[valid] - sqrt_d[valid]) / (2 * a[valid])
        solution_2[valid] = (-b[valid] + sqrt_d[valid]) / (2 * a[valid])
        solution_1 = np.where(solution_1 <= m_cutoff, float('inf'), solution_1)
        solution_2 = np.where(solution_2 <= m_cutoff, float('inf'), solution_2)
        if to_ignore >= 0:
            solution_1[to_ignore] = float('inf')
            solution_2[to_ignore] = float('inf')

        min_index_1 = np.argmin(solution_1)
        new_min_1 = solution_1.flat[min_index_1]
        min_index_2 = np.argmin(solution_2)
        new_min_2 = solution_2.flat[min_index_2]
        if new_min_1 < new_min_2:
            return min_index_1, new_min_1
        else:
            return min_index_2, new_min_2

    def calculate_molecule_interaction(self, molecule_id: int, id_to_ignore: int = None, shift: float = 0):
        cutoff_time = self.molecules_pos_time[molecule_id] + shift #- 1e-8

        min_time = float('inf')
        min_id = None
        is_border = True
        for border_id in range(-1, -self.borders_count - 1, -1):
            time = self.calculate_border_collision_time(molecule_id, border_id)
            if min_time > time > cutoff_time and id_to_ignore != border_id:
                min_time = time
                min_id = border_id

        min_index, new_min = Simulation.calculate_molecule_interaction_with_m(
            self.molecules_pos,
            self.molecules_vel,
            self.molecules_radius,
            self.molecules_pos_time,
            molecule_id,
            cutoff_time,
            self.molecules_count, id_to_ignore if id_to_ignore is not None else -1
        )

        if new_min < min_time:
            min_time = new_min
            min_id = min_index
            is_border = False

        if min_id is None:
            print(f"Error in calculating interaction, backup protocol ignored, molecule with id: {molecule_id} is erased from queue")
            return
        self.molecules_closest_interaction_time[molecule_id] = min_time
        self.molecules_queue_presence[molecule_id] += 1
        if is_border:
            self.interaction_queue.put((
                min_time,
                molecule_id,
                self.molecules_history_id[molecule_id],
                min_id,
                self.borders_history_id[min_id]
            ))
        else:
            self.molecules_queue_presence[min_id] += 1
            self.interaction_queue.put((
                min_time,
                molecule_id,
                self.molecules_history_id[molecule_id],
                min_id,
                self.molecules_history_id[min_id]
            ))

    def synch_molecule_pair(self, molecule_initiator_id: int, ignore_pair: int = -1):
        pass

    def update_molecule(
            self,
            molecule_id: int,
            new_velocity: np.ndarray[Tuple[Literal[3]],
            np.dtype[np.float64]],
            time_shift: float,
            pair_to_ignore: int = -1
    ):
        self.molecules_pos[molecule_id] += self.molecules_vel[molecule_id] * time_shift
        self.molecules_vel[molecule_id] = new_velocity
        self.molecules_pos_time[molecule_id] = self.time_since_start
        self.molecules_history_id[molecule_id] += 1
        self.synch_molecule_pair(molecule_id, pair_to_ignore)
        self.calculate_molecule_interaction(molecule_id, pair_to_ignore)

    def force_molecule_speed(self, molecule_id: int, speed: float):
        self.update_molecule(molecule_id, polar_to_cartesian(speed, np.random.random() * 2 * np.pi, np.random.random() * np.pi).reshape(3,), 0)

    # if speed positive volume reduces, if negative increases
    def force_border_speed(self, border_id: int, speed: float):
        speed *= (-1 if border_id % 2 == 1 else 1)
        self.borders_pos[border_id] += self.borders_vel[border_id] * (self.time_since_start - self.borders_pos_time[border_id])
        self.borders_vel[border_id] = speed
        self.borders_pos_time[border_id]  = self.time_since_start
        self.borders_history_id[border_id] += 1
        for molecule_index in range(0, self.molecules_count):
            time = self.calculate_border_collision_time(molecule_index, border_id)
            current_time = self.molecules_closest_interaction_time[molecule_index]
            if time is not None and time > self.time_since_start and (current_time is None or current_time > time):
                self.molecules_queue_presence[molecule_index] += 1
                self.interaction_queue.put((
                    time,
                    molecule_index,
                    self.molecules_history_id[molecule_index],
                    positive_index_to_negative(border_id, self.borders_count),
                    self.borders_history_id[border_id]
                ))

    # temperature in Kelvins
    # if None than it will stop affecting molecules
    def force_border_temperature(self, border_id: int, K: float | None):
        self.borders_temperature[border_id] = K

    def set_border_temperature(self, border_id: int, temperature: float):
        self.borders_temperature[border_id] = temperature

    def validate_interaction(self, entity_id, history_id):
        if entity_id < 0:
            return self.borders_history_id[entity_id] == history_id
        else:
            return self.molecules_history_id[entity_id] == history_id

    def make_iteration(self):
        self.last_iteration_border_momentum_effect = 0
        interaction = self.interaction_queue.get()
        time = interaction[0]
        entity_id_1 = interaction[1]
        entity_id_2 = interaction[3]
        if self.validate_interaction(entity_id_1, interaction[2]) \
                and self.validate_interaction(entity_id_2, interaction[4]):
            self.time_since_start = max(self.time_since_start, time)
            if (entity_id_1 >= 0) ^ (entity_id_2 >= 0):
                # one of them is a border
                border_id = entity_id_1 if entity_id_1 < 0 else entity_id_2
                molecule_id = entity_id_1 if entity_id_1 >= 0 else entity_id_2
                # print(f"border {border_id} by {molecule_id} with {self.molecules_history_id[molecule_id]} at {interaction[0]}")
                temp = self.borders_temperature[border_id]
                new_vel = one_sided_elastic_collision(
                    self.molecules_vel[molecule_id],
                    self.borders_normal[border_id].value.normal * (-1 if border_id % 2 == 1 else 1),
                    self.borders_vel[border_id] * (-1 if border_id % 2 == 1 else 1),
                    None if temp is None else (np.sqrt(3 * boltzmann_constant * temp / self.molecules_weight[molecule_id]))
                )
                self.last_iteration_border_momentum_effect = np.linalg.norm(self.molecules_vel[molecule_id] - new_vel) * self.molecules_weight[molecule_id]
                self.update_molecule(molecule_id, new_vel, time - self.molecules_pos_time[molecule_id], pair_to_ignore=border_id)
            else:
                # both are molecules
                # print(f"molecules by {entity_id_1} {self.molecules_history_id[entity_id_1]} and {entity_id_2} {self.molecules_history_id[entity_id_2]} at {interaction[0]}")
                new_vel_1, new_vel_2 = elastic_balls_interaction(
                    self.molecules_pos[entity_id_1],
                    self.molecules_vel[entity_id_1],
                    self.molecules_weight[entity_id_1],
                    self.molecules_pos[entity_id_2],
                    self.molecules_vel[entity_id_2],
                    self.molecules_weight[entity_id_2],
                )
                self.update_molecule(entity_id_1, new_vel_1, time - self.molecules_pos_time[entity_id_1], pair_to_ignore=entity_id_2)
                self.update_molecule(entity_id_2, new_vel_2, time - self.molecules_pos_time[entity_id_2], pair_to_ignore=entity_id_1)
        for id in [entity_id_1, entity_id_2]:
            if id >= 0:
                self.molecules_queue_presence[id] -= 1
                # pos = self.molecules_pos[id] + self.molecules_vel[id] * (self.time_since_start - self.molecules_pos_time[id])
                # if pos[0] < self.borders_pos[0] or pos[0] > self.borders_pos[1] or pos[1] < self.borders_pos[2] or pos[1] > self.borders_pos[3] or pos[2] < self.borders_pos[4] or pos[2] > self.borders_pos[5]:
                #     self.molecules_pos[id] = self.borders_pos[1:, 2] - self.borders_pos[:, 2]
                if self.molecules_queue_presence[id] == 0:
                    self.calculate_molecule_interaction(id)


    def get_current_positions(self):
        return self.molecules_pos + self.molecules_vel * ((self.time_since_start - self.molecules_pos_time)[:, np.newaxis])

    def get_current_border_positions(self):
        return self.borders_pos + self.borders_vel * (self.time_since_start - self.borders_pos_time)

    def get_current_volume(self):
        positions = self.get_current_border_positions()
        side1 = positions[1] - positions[0]
        side2 = positions[3] - positions[2]
        side3 = positions[5] - positions[4]
        return side1 * side2 * side3

    def get_current_area(self):
        positions = self.get_current_border_positions()
        side1 = positions[1] - positions[0]
        side2 = positions[3] - positions[2]
        side3 = positions[5] - positions[4]
        return 2 * (side1 * side2 + side2 * side3 + side1 * side3)
