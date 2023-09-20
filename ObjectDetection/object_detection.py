from typing import Dict
from abc import ABC


import cv2
import torch
from torch import Tensor
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

class BoundingBoxes:

    def __init__(self, boxes: Tensor, orig_shape: tuple[int], names: Dict[int,str]):
        if boxes.ndim == 1:
            boxes = boxes[None, :]

        self.data = boxes
        self.names = names
        self.orig_shape = orig_shape

    def cpu(self):
        """Return a copy of the tensor on CPU memory."""
        return self.__class__(self.data.cpu(), self.orig_shape, self.names)

    def cuda(self):
        """Return a copy of the tensor on GPU memory."""
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape, self.names)

    def __len__(self):  # override len(results)
        """Return the length of the data tensor."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a BaseTensor with the specified index of the data tensor."""
        return self.__class__(self.data[idx], self.orig_shape, self.names)

    @property
    def xyxy(self) -> Tensor:
        """Return the boxes in xyxy format."""
        return self.data[:, :4]

    @property
    def conf(self) -> Tensor:
        """Return the confidence values of the boxes."""
        return self.data[:, -2]

    @property
    def cls(self) -> Tensor:
        """Return the class values of the boxes."""
        return self.data[:, -1]

class ObjectDetection(ABC):
    def __init__(self, model_path: str):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using Device: {self.device}')
        
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path: str):
        pass

    def predict(self,frame: np.ndarray) -> BoundingBoxes:
        pass
    
class Yolo(ObjectDetection):
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
    
    def load_model(self, model_path) -> YOLO:
        model = YOLO(model_path)
        model.fuse()
        return model
    
    def predict(self,frame: np.ndarray) -> BoundingBoxes:
        preds = self.model(frame)[0]

        result = BoundingBoxes(preds.boxes.data, frame.shape[:2], preds.names)

        if self.device == 'cuda':
            result.cuda()
        else:
            result.cpu()

        return result
    
