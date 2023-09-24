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
    

class Colors:

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class Annotator:
    
    def __init__(self, im):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.im = im
        self.lw = max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128)):
        """Add one xyxy box to image with label."""
        if isinstance(box, Tensor):
            box = box.tolist()

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        self.lw / 3,
                        (255, 255, 255),
                        thickness=tf,
                        lineType=cv2.LINE_AA)

colors = Colors()


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
    
    def plot_boxes(self, img: np.ndarray, pred_boxes: BoundingBoxes) -> np.ndarray:
        names = pred_boxes.names
        annotator = Annotator(img)

        if pred_boxes:
            for d in reversed(pred_boxes):
                c, conf = int(d.cls), float(d.conf)
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

        return annotator.im
