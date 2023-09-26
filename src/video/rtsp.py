from src.video.stream_thread import VideoThread
from queue import Queue
import cv2
import os

class VideoRTSP(VideoThread):
    def __init__(self, rstp_url: str, width: int = None, height: int = None, queueSize : int = 126):
        self.rstp_url = rstp_url
        self.grabbed = None
        self.frame = None
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)
        self.size = (width,height)
        
        self.stream = cv2.VideoCapture(self.rstp_url)
        self.opened = self.stream.isOpened()
        if (not self.opened):
            raise RuntimeError('Error opening the video rtsp')
        self.start()


