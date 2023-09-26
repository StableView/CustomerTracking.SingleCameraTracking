from abc import ABC, abstractmethod

class FeatureExtractorBase(ABC):
    @abstractmethod
    def extract_features(self, frame):
        pass