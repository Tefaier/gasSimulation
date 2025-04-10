from enum import Enum
from typing import Tuple, Literal

import numpy as np

class Vector:
    def __init__(self, id: int, vector_form: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]):
        self.id = id
        self.normal = vector_form


class Axis(Enum):
    x = Vector(0, np.array([1, 0, 0], dtype=np.float64))
    y = Vector(1, np.array([0, 1, 0], dtype=np.float64))
    z = Vector(2, np.array([0, 0, 1], dtype=np.float64))
