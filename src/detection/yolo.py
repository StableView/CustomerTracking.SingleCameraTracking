"""Yolo implementations for detection"""
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes


from src.detection.object_detection import ObjectDetection
from src.detection.boxes import BoundingBoxes


class Yolov8(ObjectDetection):
    """Yolov8 sub-class"""

    def load_model(self, model_path) -> YOLO:
        """Load YoloV8 object"""
        model = YOLO(model_path)
        model.fuse()
        return model

    def predict(self,frame: np.ndarray) -> np.ndarray:
        """Inference return a np array"""
        preds = self.model(frame)[0]
        result = preds.boxes.data.clone()
        result = result[:, [5, 4, 0, 1, 2, 3]]
        result = result[result[:, 0] == 0.0]
        #result = torch.jit._unwrap_optional(result)
        if self.device == 'cuda':
            result.cuda()
        else:
            result.cpu()

        return result.numpy()

    def predict_to_boundingboxes(self,frame: np.ndarray) -> BoundingBoxes:
        """Inference return a BoundingBoxes object"""
        result = self.predict(frame)
        return BoundingBoxes(frame, result, self.model.model.names)
