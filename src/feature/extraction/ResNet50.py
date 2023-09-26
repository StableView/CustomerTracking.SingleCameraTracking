from PIL import Image
from img2vec_pytorch import Img2Vec
from src.detection.detection_msg import DetectionMsg
from src.feature.extraction.feature_extractor import FeatureExtractorBase


class ResNet50FeatureExtractor(FeatureExtractorBase):
    def __init__(self) -> None:
        super().__init__()
        self.img2vec = Img2Vec(cuda=False, model='resnet-18')

    def extract_features(self, frame: DetectionMsg):
        output = []
        for bbox in frame.boxes:
            img = Image.fromarray(frame.image[int(bbox[2]):int(bbox[3]), int(bbox[4]):int(bbox[5])])
            result = self.img2vec.get_vec(img, tensor=False)
            output.append(result)
        return output
