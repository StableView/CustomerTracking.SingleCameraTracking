from src.video.stream_thread import VideoThread
from queue import Queue
import cv2

class VideoCamera(VideoThread):
    def __init__(self, src: int = 0, width: int = None, height: int = None, queueSize : int = 126):
        self.src = src
        self.grabbed = None
        self.frame = None
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)
        self.size = (width,height)
        
        self.stream = cv2.VideoCapture(self.src)
        self.opened = self.stream.isOpened()
        if (not self.opened):
            raise RuntimeError('Error opening the video camera')
        self.start()


