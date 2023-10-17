import numpy as np
import torch

def xyxy2tlwh(x):
    """
    Convert bounding box coordinates from (t, l ,w ,h) format to (t, l, w, h) format where (t, l) is the
    top-left corner and (w, h) is width and height.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def xyxy2xyah(x):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    y = np.copy(x)
    y[:2] += y[2:] / 2
    y[2] /= y[3]
    return y