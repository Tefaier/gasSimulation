from typing import Any, Tuple, Literal

import numpy as np

boltzmann_constant = 1.380649e-23

def cartesian_product(x, y):  # makes array with (y_size, x_size, 2, 3)
    dim_x = len(x)
    dim_y = len(y)
    if len(x.shape) == 1:
        x_r = np.tile(x, dim_y).reshape((dim_y, dim_x, 1))
        y_r = np.repeat(y, dim_x, axis=0).reshape((dim_y, dim_x, 1))
        return np.concatenate([x_r, y_r], axis=2)
    dim_info = x.shape[-1]
    x_r = np.tile(x, (dim_y, 1)).reshape((dim_y, dim_x, dim_info))
    y_r = np.repeat(y, dim_x, axis=0).reshape((dim_y, dim_x, dim_info))
    return np.concatenate([x_r, y_r], axis=2).reshape((dim_y, dim_x, 2, dim_info))

def polar_to_cartesian(r: np.ndarray[Any, np.dtype[np.float64]], theta: np.ndarray[Any, np.dtype[np.float64]], ksi: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    if type(r) is float:
        r = np.array([r])
        theta = np.array([theta])
        ksi = np.array([ksi])

    result = np.zeros((*r.shape, 3))

    result[:, 0] = np.sin(ksi) * np.cos(theta)
    result[:, 1] = np.sin(ksi) * np.sin(theta)
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
    contact_normal = vec_normalize(pos_1 - pos_2)
    v_normal = np.dot(vel_1 - vel_2, contact_normal) * contact_normal
    new_vel_1 = vel_1 - v_normal * 2 * weight_2 / (weight_1 + weight_2)
    new_vel_2 = vel_2 + v_normal * 2 * weight_1 / (weight_1 + weight_2)
    # m_reduced = 1 / (1 / weight_1 + 1 / weight_2)
    # impact_vel = np.dot(contact_normal, vel_1 - vel_2)
    # impulse_shift = 2 * m_reduced * impact_vel
    # new_vel_1 = vel_1 - impulse_shift * contact_normal / weight_1
    # new_vel_2 = vel_2 + impulse_shift * contact_normal / weight_2
    return new_vel_1, new_vel_2

def one_sided_elastic_collision(
        vel: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]],
        contact_normal: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]],  # normal must look into bounce area
        contact_speed: float = 0,  # if positive increases bounce, if negative reduces bounce
        absolute_speed_force: float = None  # should be used for temperature to enforce speed after collision, for now if present, contact_speed must be 0
) -> np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]:
    in_normal_speed = np.dot(contact_normal, vel)
    # check that bounce is impossible
    if contact_speed < in_normal_speed:
        return vel
    new_velocity = vel - ((contact_normal * in_normal_speed * 2) if in_normal_speed < 0 else 0)  # just rebounce
    if absolute_speed_force is not None:
        new_velocity = vec_normalize(new_velocity) * absolute_speed_force
    new_velocity += contact_normal * contact_speed  # modification of velocity according to border movement
    return new_velocity

def positive_index_to_negative(index: int, size: int) -> int:
    return index - size
