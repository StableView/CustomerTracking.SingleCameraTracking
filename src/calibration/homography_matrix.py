import cv2
import numpy as np
from src.calibration.calibration import CameraCalibration

class HomographyMatrix(CameraCalibration):
    def __init__(self) -> None:
        self.points_reference: np.array = None
        self.points_camera: np.array = None
        self.homography_matrix: np.array = None
        
    def calibrate(self,points_reference: np.array,points_camera:np.array) -> np.array:
        self.points_reference = points_reference
        self.points_camera = points_camera

        if (len(self.points_camera) == len(self.points_reference)) and  (len(self.points_reference) >=4):
            self.homography_matrix, _ = cv2.findHomography(self.points_camera,self.points_reference)
        else:
            self.homography_matrix = None

        return self.homography_matrix