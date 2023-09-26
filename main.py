import cv2
import numpy as np
from src.detection.detection_msg import DetectionMsg
from src.feature.controller import FeatureController
from src.feature.extraction.mp_pose import MPPoseFeatureExtractor
from src.feature.extraction.resnet50 import ResNet50FeatureExtractor
import logging


# Configura el sistema de registro
log_filename = 'app.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    detectionLst = [
        DetectionMsg(cv2.imread('src/data/eg_CASIA_bag.png'),  np.array([[1, 0.9, 50, 200, 120, 210 ]])),
        DetectionMsg(cv2.imread('src/data/eg_CASIA_coat.png'), np.array([[1, 0.9, 45, 200, 135, 210 ]])),
        DetectionMsg(
            cv2.imread('src/data/people.png'), 
            np.array([
                [1, 0.6, 10, 170,   0,  50], 
                [2, 0.9,  5, 185,  45, 120], 
                [3, 0.9, 40, 215, 125, 215],
                [4, 0.9, 40, 205, 190, 275]
            ])
        )
    ]


    feature_extractors = {
        'mppose':   MPPoseFeatureExtractor(),
        'resnet50': ResNet50FeatureExtractor()
    }
    feature_controller = FeatureController(feature_extractors)
    features = feature_controller.extract_features_from_objects(detectionLst[2])
    
    logging.info("")
    logging.info("Aplicacion iniciada")
    logging.info("")
    logging.info("MediaPipe Pose")
    for ii in features['mppose']:
        logging.info(len(ii))
        logging.info('-' * 10)
    logging.info("")
    logging.info("ResNet50")
    for ii in features['resnet50']:
        logging.info(ii.shape)
        logging.info('-' * 10)
    logging.info("")
    logging.info("Aplicacion finalizada")
