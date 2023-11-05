import numpy as np
from statistics import mean 
from src.feature.position_msg import PositionMsg
from src.feature.feet.position_extractor import PositionExtractorBase

LANDMARKS = 33
THR_VISIBILITY = 0.5
# https://es.canson.com/consejos-de-expertos/dibujo-las-proporciones-humanas
PRG_HIP = 1 + 4 / 7
PRG_KNEE = 1 + 2 / 7

class MPPosePositionExtractor(PositionExtractorBase):
    def __init__(self) -> None:
        super().__init__()

    def extract_features(self, frame: list):
        output = PositionMsg(boxes=[], type='Normal')

        if (len(frame) != LANDMARKS):
            output.type = 'Bad'
            return output
        else:
            visibility_left_shoulder = frame[11][3] > THR_VISIBILITY
            visibility_right_shoulder = frame[12][3] > THR_VISIBILITY
            visibility_legs = not any(landmark[3] < THR_VISIBILITY for landmark in frame[22:32])

            if not (visibility_legs and visibility_left_shoulder and visibility_right_shoulder):
                output.type = 'Estimated'

            # ankle
            if (frame[27][3] > THR_VISIBILITY and frame[28][3] > THR_VISIBILITY):
                output.type = 'Normal'
                output.boxes = np.array([mean([frame[27][0], frame[28][0]]), mean([frame[27][1], frame[28][1]])])
                return output
            
            # knee
            if (frame[25][3] > THR_VISIBILITY and frame[26][3] > THR_VISIBILITY):
                output.boxes = np.array([mean([frame[25][0], frame[26][0]]), mean([frame[25][1], frame[26][1]]) * PRG_KNEE])
                return output

            # hip
            if (frame[23][3] > THR_VISIBILITY and frame[24][3] > THR_VISIBILITY):
                output.boxes = np.array([mean([frame[23][0], frame[24][0]]), mean([frame[23][1], frame[24][1]]) * PRG_HIP])
                return output

            return output

 