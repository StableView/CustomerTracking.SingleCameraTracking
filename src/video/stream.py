
from abc import ABC, abstractstaticmethod, abstractmethod

class VideoStream(ABC):

    @abstractmethod
    def read(self):
        pass
    
    @abstractmethod
    def is_opened(self):
        pass

    @abstractmethod
    def stop(self):
        pass
