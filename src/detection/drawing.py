"""Draw bounding boxes module"""
import cv2
from torch import Tensor
import numpy as np

from src.detection.boxes import BoundingBoxes


class Colors:
    """Pallete Color"""
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A',
                '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF',
                '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF',
                'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.len_pallete = len(self.palette)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        rgb_color = self.palette[int(i) % self.len_pallete]
        return (rgb_color[2], rgb_color[1], rgb_color[0]) if bgr else rgb_color

    @staticmethod
    def hex2rgb(hex_color):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(hex_color[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class Annotator:
    """Draw each bounding box"""
    def __init__(self, image):
        self.image = image
        self.line_width = max(round(sum(image.shape) / 2 * 0.003), 2)

    def box_label(self, box, label='', color=(128, 128, 128)):
        """Add one xyxy box to image with label."""
        if isinstance(box, Tensor):
            box = box.tolist()

        point_1, point_2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.image, point_1, point_2, color, thickness=self.line_width,
                      lineType=cv2.LINE_AA)
        if label:
            thick_font = max(self.line_width - 1, 1)
            text_width, text_height= cv2.getTextSize(label, 0, fontScale=self.line_width / 3,
                                                     thickness=thick_font)[0]
            outside = point_1[1] - text_height>= 3
            point_2 = point_1[0] + text_width, point_1[1] - text_height- 3\
                      if outside else point_1[1] + text_height+ 3
            cv2.rectangle(self.image, point_1, point_2, color, -1, cv2.LINE_AA)
            cv2.putText(self.image,
                        label, (point_1[0], point_1[1] - 2\
                                if outside else point_1[1] + text_height+ 2),
                        0,
                        self.line_width / 3,
                        (255, 255, 255),
                        thickness=thick_font,
                        lineType=cv2.LINE_AA)

colors = Colors()

class Draw():
    """Drawing class for bounding boxes"""
    @staticmethod
    def plot_boxes(pred_boxes: BoundingBoxes) -> np.ndarray:
        """Plot Predictions"""
        names = pred_boxes.names
        annotator = Annotator(pred_boxes.orig_img)

        if pred_boxes:
            for pred in reversed(pred_boxes):
                pred_class, conf = int(pred.cls), float(pred.conf)
                label = f'{names[pred_class]} {conf:.2f}'
                annotator.box_label(pred.xyxy.squeeze(), label, color=colors(pred_class, True))

        return annotator.image
    