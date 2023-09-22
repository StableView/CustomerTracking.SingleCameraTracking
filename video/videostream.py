
from abc import ABC, abstractstaticmethod, abstractmethod

class VideoStream(ABC):

    @abstractmethod
    def read(self):
        pass
    
    @abstractmethod
    def isOpened(self):
        pass

    @abstractmethod
    def stop(self):
        pass
