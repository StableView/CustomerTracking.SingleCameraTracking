"""object detection module"""
from abc import ABC, abstractmethod
import torch
import numpy as np


class ObjectDetection(ABC):
    """Base class for object detectector"""
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using Device: {self.device}')
        self.model = self.load_model(model_path)

    @abstractmethod
    def load_model(self, model_path: str):
        """Load the model object"""

    @abstractmethod
    def predict(self,frame: np.ndarray) -> np.array:
        """Run inference return an array where each row is: class,conf,x,y,x,y"""
