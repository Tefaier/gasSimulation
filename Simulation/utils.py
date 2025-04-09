from typing import Any, Tuple, Literal

import numpy as np


def polar_to_cartesian(r: np.ndarray[Any, np.dtype[np.float64]], theta: np.ndarray[Any, np.dtype[np.float64]], ksi: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    if type(r) is float:
        r = np.ndarray([r])
        theta = np.ndarray([theta])
        ksi = np.ndarray([ksi])

    result = np.zeros((*r.shape, 3))

    result[:, 0] = np.sin(ksi) * np.cos(theta[:, 1])
    result[:, 1] = np.sin(ksi) * np.sin(theta[:, 1])
    result[:, 2] = np.cos(ksi)

    result *= r[:, np.newaxis]
    return result

def vec_normalize(vec: np.array) -> np.array:
    length = np.linalg.norm(vec)
    return np.divide(vec, np.linalg.norm(vec), where=length!=0)

# returns new velocities of 1 and 2 as a tuple of two vectors
def elastic_balls_interaction(
        pos_1: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]],
        vel_1: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]],
        weight_1: float,
        pos_2: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]],
        vel_2: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]],
        weight_2: float
) -> Tuple[np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]], np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]]:
    contact_normal = vec_normalize(pos_2 - pos_1)
    m_reduced = 1 / (1 / weight_1 + 1 / weight_2)
    impact_vel = np.dot(contact_normal, vel_2 - vel_1)
    impulse_shift = 2 * m_reduced * impact_vel
    return vel_1 - impulse_shift * contact_normal / weight_1, vel_2 + impulse_shift * contact_normal / weight_2

def one_sided_elastic_collision(
        pos: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]],
        vel: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]],
        contact_normal: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]],
        contact_speed: float = 0,  # if positive increases bounce, if negative reduces bounce
        absolute_speed_force: float = None  # should be used for temperature to enforce speed after collision, for now if present, contact_speed must be 0
) -> np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]:
    in_normal_speed = abs(np.dot(contact_normal, vel))
    if contact_speed < -in_normal_speed:
        return vel
    new_velocity = vel - contact_normal * np.dot(contact_normal, vel) * (2 + contact_speed / in_normal_speed)
    if absolute_speed_force is not None:
        return vec_normalize(new_velocity) * absolute_speed_force
    return new_velocity

def positive_index_to_negative(index: int, size: int) -> int:
    pass
