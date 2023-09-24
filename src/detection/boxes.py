from typing import Dict
import numpy as np


class BoundingBoxes:

    def __init__(self, orig_img: np.ndarray, boxes: np.ndarray, names: Dict[int,str]):
        if boxes.ndim == 1:
            boxes = boxes[None, :]

        self.data = boxes
        self.names = names
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]

    def cpu(self):
        """Return a copy of the tensor on CPU memory."""
        pass
        #return self.__class__(self.data.cpu(), self.orig_shape, self.names)

    def cuda(self):
        """Return a copy of the tensor on GPU memory."""
        pass
        #return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape, self.names)

    def __len__(self):  # override len(results)
        """Return the length of the data tensor."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a BaseTensor with the specified index of the data tensor."""
        return self.__class__(self.orig_img, self.data[idx], self.names)

    @property
    def xyxy(self) -> np.ndarray:
        """Return the boxes in xyxy format."""
        return self.data[:, 2:]

    @property
    def conf(self) -> np.ndarray:
        """Return the confidence values of the boxes."""
        return self.data[:, 1]

    @property
    def cls(self) -> np.ndarray:
        """Return the class values of the boxes."""
        return self.data[:, 0]
