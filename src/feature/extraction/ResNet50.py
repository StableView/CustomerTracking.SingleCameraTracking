from PIL import Image
from img2vec_pytorch import Img2Vec
from src.detection.DetectionMsg import DetectionMsg
from src.feature.extraction.FeatureExtractor import FeatureExtractorBase


class ResNet50FeatureExtractor(FeatureExtractorBase):
    def __init__(self) -> None:
        super().__init__()
        self.img2vec = Img2Vec(cuda=False, model='resnet-18')

    def extract_features(self, frame: DetectionMsg):
        output = []
        for bbox in frame.boxes:
            img = Image.fromarray(frame.image[bbox[0]:bbox[1], bbox[2]:bbox[3]])
            result = self.img2vec.get_vec(img, tensor=False)
            output.append(result)
        return output
