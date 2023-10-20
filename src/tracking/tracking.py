""" tracking module """
from abc import ABC, abstractmethod
import numpy as np

class Tracking(ABC):
    """Base class for tracking"""
    def __init__(self):
        self.tracks = None

    @abstractmethod
    def update(self, detections: np.ndarray):
        """Main method, update the tracks"""

    @abstractmethod
    def predict_detection(self):
        """predict state for detections"""

    @abstractmethod
    def associate(self):
        """Make the association between tracks and detections"""