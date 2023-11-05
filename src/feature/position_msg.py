import numpy as np
from dataclasses import dataclass

@dataclass
class PositionMsg:
    boxes: np.ndarray
    type: str