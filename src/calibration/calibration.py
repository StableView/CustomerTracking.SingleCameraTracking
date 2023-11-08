from abc import ABC, abstractmethod

class CameraCalibration(ABC):

    @abstractmethod
    def calibrate(self):
        pass