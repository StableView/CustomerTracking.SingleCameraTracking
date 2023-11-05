from abc import ABC, abstractmethod

class PositionExtractorBase(ABC):
    @abstractmethod
    def extract_features(self, frame):
        pass