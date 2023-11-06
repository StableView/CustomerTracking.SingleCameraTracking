import numpy as np
from enum import Enum
from dataclasses import dataclass

@dataclass
class PositionMsg:
    boxes: np.ndarray
    type: "PositionType"

class PositionType(str, Enum):
    BAD = "Bad"
    NORMAL = "Normal"
    ESTIMATED = "Estimated"