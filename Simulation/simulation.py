from typing import Tuple, Any, Literal

import numpy as np
from queue import PriorityQueue

from Simulation.models import Axis
from Simulation.utils import polar_to_cartesian, elastic_balls_interaction, one_sided_elastic_collision, \
    positive_index_to_negative, cartesian_product


class Simulation:
    time_since_start: float

    molecules_count: int
    molecules_radius: np.ndarray[Any, np.dtype[np.float64]]
    molecules_weight: np.ndarray[Any, np.dtype[np.float64]]
    molecules_pos: np.ndarray[Any, np.dtype[np.float64]]
    molecules_vel: np.ndarray[Any, np.dtype[np.float64]]
    molecules_pos_time: np.ndarray[Any, np.dtype[np.float64]]
    molecules_history_id: list[int]
    molecules_expected_pair: list[int]  # if -1 then it is not a molecule or there is nothing at all
    molecules_closest_interaction_time: list[float | None]

    borders_count: int
    borders_normal: list[Axis] = [Axis.x, Axis.x, Axis.y, Axis.y, Axis.z, Axis.z]
    borders_pos: list[float]
    borders_vel: list[float]
    borders_pos_time: list[float]
    borders_history_id: list[int]
    borders_temperature: list[float | None]

    #  object inside is as follows:
    #  time of interaction
    #  id of first interacting entity
    #  history_id of first entity when interaction was added to queue
    #  id of second interacting entity
    #  history_id of second entity when interaction was added to queue
    interaction_queue: PriorityQueue[Tuple[float, int, int, int, int]]



    def __init__(self, n: int, molecule_radius: float, molecules_weight: float, initial_max_speed: float, initial_volume: float):
        self.molecules_count = n
        self.time_since_start = 0
        side_length = initial_volume ** (1/3)
        self.init_molecules(molecule_radius, molecules_weight, side_length * 0.5 - molecule_radius * 1.1, initial_max_speed)
        self.init_borders(side_length / 2)
        self.interaction_queue = PriorityQueue()
        for molecule_id in range(0, n):
            self.calculate_molecule_interaction(molecule_id)

    def init_molecules(self, radius: float, weight: float, max_offset: float, max_speed: float):
        self.molecules_radius = np.ones(shape=(self.molecules_count,)) * radius
        self.molecules_weight = np.ones(shape=(self.molecules_count,)) * weight
        while True:
            self.molecules_pos = np.random.rand(self.molecules_count, 3) * 2 * max_offset - max_offset
            cross = cartesian_product(self.molecules_pos, self.molecules_pos)
            cross[np.eye(self.molecules_count, self.molecules_count) > 0.5, 0, :] = 1e10
            min_dist_2 = np.min(np.sum(((cross[:, :, 0, :] - cross[:, :, 1, :]) ** 2), axis=2))
            if min_dist_2 > self.molecules_radius[0]**2:
                break
        self.molecules_vel = polar_to_cartesian(
            np.random.rand(self.molecules_count) * max_speed,
            np.random.rand(self.molecules_count) * 2 * np.pi,
            np.random.rand(self.molecules_count) * np.pi
        )
        self.molecules_pos_time = np.zeros(shape=(self.molecules_count,))
        self.molecules_history_id = [0] * self.molecules_count
        self.molecules_expected_pair = [-1] * self.molecules_count
        self.molecules_closest_interaction_time = [0] * self.molecules_count

    def init_borders(self, offset: float):
        self.borders_count = len(self.borders_normal)

        self.borders_pos = [-offset, offset] * 3
        self.borders_vel = [0] * self.borders_count
        self.borders_pos_time = [0] * self.borders_count
        self.borders_history_id = [0] * self.borders_count
        self.borders_temperature = [None] * self.borders_count

    def calculate_border_collision_time(self, molecule_id: int, border_id: int) -> float | None:
        axis_mask = self.borders_normal[border_id].value.id
        speed_diff = self.borders_vel[border_id] - self.molecules_vel[molecule_id][axis_mask]
        if abs(speed_diff) < 1e-8: return None
        return (
                (self.molecules_radius[molecule_id] * (1 if border_id % 2 == 1 else -1)
                   + self.molecules_pos[molecule_id][axis_mask]
                   - self.borders_pos[border_id]
                   - self.molecules_vel[molecule_id][axis_mask] * self.molecules_pos_time[molecule_id]
                   + self.borders_vel[border_id] * self.borders_pos_time[border_id])
                / speed_diff
        )

    def calculate_molecule_interaction(self, molecule_id: int):
        min_time = 0
        min_id = None
        is_border = True
        for border_id in range(0, self.borders_count):
            time = self.calculate_border_collision_time(molecule_id, border_id)
            if time > self.time_since_start and (min_id is None or min_time > time):
                min_time = time
                min_id = border_id

        pos = self.molecules_pos[molecule_id]
        vel = self.molecules_vel[molecule_id]
        rad = self.molecules_radius[molecule_id]
        time = self.molecules_pos_time[molecule_id]
        # a * t**2 + b * t + c = 0
        a = np.sum((vel[np.newaxis, :] - self.molecules_vel) ** 2, axis=1)
        b = np.sum(2 * (self.molecules_vel - vel[np.newaxis, :]) * (self.molecules_pos - pos[np.newaxis, :] + time * vel[np.newaxis, :] - self.molecules_pos_time[:, np.newaxis] * self.molecules_vel), axis=1)
        c =  - ((rad + self.molecules_radius) ** 2) + np.sum((time * vel[np.newaxis, :] - self.molecules_pos_time[:, np.newaxis] * self.molecules_vel + self.molecules_pos - pos[np.newaxis, :])**2, axis=1)
        d = b ** 2 - 4 * a * c
        solutions = np.concatenate([((-b-np.sqrt(d))/(2 * a)).reshape((self.molecules_count, 1)), ((-b+np.sqrt(d))/(2 * a)).reshape((self.molecules_count, 1))], axis=1)
        solutions[d <= 0, :] = 1e10
        solutions[solutions < self.time_since_start + 1e-8] = 1e10
        min_index = np.unravel_index(np.argmin(solutions, axis=None), solutions.shape)
        new_min = solutions[min_index]
        if new_min < min_time:
            min_time = new_min
            min_id = min_index[0]
            is_border = False
        self.molecules_closest_interaction_time[molecule_id] = min_time
        if is_border:
            self.molecules_expected_pair[molecule_id] = -1
            self.interaction_queue.put((
                min_time,
                molecule_id,
                self.molecules_history_id[molecule_id],
                positive_index_to_negative(min_id, self.borders_count),
                self.borders_history_id[min_id]
            ))
        else:
            self.molecules_expected_pair[molecule_id] = min_id
            self.interaction_queue.put((
                min_time,
                molecule_id,
                self.molecules_history_id[molecule_id],
                min_id,
                self.molecules_history_id[min_id]
            ))

    def synch_molecule_pair(self, molecule_initiator_id: int, ignore_pair: int = -1):
        paired_id = self.molecules_expected_pair[molecule_initiator_id]
        if not paired_id == -1 and self.molecules_expected_pair[paired_id] == molecule_initiator_id and not paired_id == ignore_pair:
            self.molecules_expected_pair[paired_id] = -1
            self.calculate_molecule_interaction(paired_id)

    def update_molecule(
            self,
            molecule_id: int,
            new_velocity: np.ndarray[Tuple[Literal[3]],
            np.dtype[np.float64]],
            time_shift: float,
            jump_value: float = 0,
            paired_id: int = -1
    ):
        self.molecules_pos[molecule_id] += self.molecules_vel[molecule_id] * time_shift + new_velocity * jump_value
        self.molecules_vel[molecule_id] = new_velocity
        self.molecules_pos_time[molecule_id] += time_shift
        self.molecules_history_id[molecule_id] += 1
        self.synch_molecule_pair(molecule_id, paired_id)
        self.calculate_molecule_interaction(molecule_id)

    def force_molecule_speed(self, molecule_id: int, speed: float):
        self.update_molecule(molecule_id, polar_to_cartesian(speed, np.random.random() * 2 * np.pi, np.random.random() * np.pi).reshape(3,), 0)

    def force_border_speed(self, border_id: int, speed: float):
        self.borders_pos[border_id] += self.borders_vel[border_id] * (self.time_since_start - self.borders_pos_time[border_id])
        self.borders_vel[border_id] = speed
        self.borders_pos_time[border_id]  = self.time_since_start
        self.borders_history_id[border_id] += 1
        for molecule_index in range(0, self.molecules_count):
            time = self.calculate_border_collision_time(molecule_index, border_id)
            current_time = self.molecules_closest_interaction_time[molecule_index]
            if time is not None and time > self.time_since_start and (current_time is None or current_time > time):
                self.interaction_queue.put((
                    time,
                    molecule_index,
                    self.molecules_history_id[molecule_index],
                    positive_index_to_negative(border_id, self.borders_count),
                    self.borders_history_id[molecule_index]
                ))

    def set_border_temperature(self, border_id: int, temperature: float):
        self.borders_temperature[border_id] = temperature

    def validate_interaction(self, entity_id, history_id):
        if entity_id < 0:
            return self.borders_history_id[entity_id] == history_id
        else:
            return self.molecules_history_id[entity_id] == history_id

    def make_iteration(self):
        interaction = self.interaction_queue.get()
        time = interaction[0]
        self.time_since_start = time
        entity_id_1 = interaction[1]
        entity_id_2 = interaction[3]
        if self.validate_interaction(entity_id_1, interaction[2]) \
                and self.validate_interaction(entity_id_2, interaction[4]):
            if (entity_id_1 >= 0) ^ (entity_id_2 >= 0):
                # one of them is a border
                border_id = entity_id_1 if entity_id_1 < 0 else entity_id_2
                molecule_id = entity_id_1 if entity_id_1 >= 0 else entity_id_2
                new_vel = one_sided_elastic_collision(
                    self.molecules_pos[molecule_id],
                    self.molecules_vel[molecule_id],
                    self.borders_normal[border_id].value.normal,
                    self.borders_vel[border_id]
                    # TODO add support for border temperature here with enforcing speed
                )
                self.update_molecule(molecule_id, new_vel, time - self.molecules_pos_time[molecule_id], 1e-8)
            else:
                # both are molecules
                new_vel_1, new_vel_2 = elastic_balls_interaction(
                    self.molecules_pos[entity_id_1],
                    self.molecules_vel[entity_id_1],
                    self.molecules_weight[entity_id_1],
                    self.molecules_pos[entity_id_2],
                    self.molecules_vel[entity_id_2],
                    self.molecules_weight[entity_id_2],
                )
                # print(f"Distance is {np.linalg.norm(self.molecules_pos[0] - self.molecules_pos[1])}")
                self.update_molecule(entity_id_1, new_vel_1, time - self.molecules_pos_time[entity_id_1], 1e-8, entity_id_2)
                self.update_molecule(entity_id_2, new_vel_2, time - self.molecules_pos_time[entity_id_2], 1e-8, entity_id_1)
        # print(f"Distance is {np.linalg.norm(self.molecules_pos[0] - self.molecules_pos[1])}")
        # print(self.interaction_queue.qsize())

    def get_current_positions(self):
        return self.molecules_pos + self.molecules_vel * ((self.time_since_start - self.molecules_pos_time)[:, np.newaxis])
