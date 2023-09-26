import mediapipe as mp
from src.detection.detection_msg import DetectionMsg
from src.feature.extraction.feature_extractor import FeatureExtractorBase

    
class MPPoseFeatureExtractor(FeatureExtractorBase):
    def __init__(self) -> None:
        super().__init__()
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def extract_features(self, frame: DetectionMsg):
        output = []
        for bbox in frame.boxes:
            img = frame.image[int(bbox[2]):int(bbox[3]), int(bbox[4]):int(bbox[5])]
            results = self.pose.process(img)
            rpose = [] if (results.pose_landmarks == None) \
                else [[ii.x, ii.y, ii.z, ii.visibility] for ii in results.pose_landmarks.landmark]
            output.append(rpose)
        return output
