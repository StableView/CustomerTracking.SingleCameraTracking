from typing import Dict
from abc import ABC


import cv2
import torch
from torch import Tensor
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
class ObjectDetection(ABC):
    def __init__(self, model_path: str):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using Device: {self.device}')
        
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path: str):
        pass

    def predict(self,frame: np.ndarray) -> BoundingBoxes:
        pass
    
