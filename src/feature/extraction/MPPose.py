import mediapipe as mp
from src.detection.DetectionMsg import DetectionMsg
from src.feature.extraction.FeatureExtractor import FeatureExtractorBase

    
class MPPoseFeatureExtractor(FeatureExtractorBase):
    def __init__(self) -> None:
        super().__init__()
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def extract_features(self, frame: DetectionMsg):
        output = []
        for bbox in frame.boxes:
            img = frame.image[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            results = self.pose.process(img)
            rpose = [] if (results.pose_landmarks == None) \
                else [[ii.x, ii.y, ii.z, ii.visibility] for ii in results.pose_landmarks.landmark]
            output.append(rpose)
        return output
