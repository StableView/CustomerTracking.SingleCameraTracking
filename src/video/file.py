from src.video.stream_thread import VideoThread
from queue import Queue
import cv2
import os

# list of different types of file
VIDEO_EXTENSIONS = [".mp4", ".avi"]

class VideoFile(VideoThread):
    def __init__(self, video_path: str, width: int = None, height: int = None, queueSize : int = 126):
        self.video_path = video_path
        self.grabbed = None
        self.frame = None
        self.stopped = False
        self.opened = False
        self.Q = Queue(maxsize=queueSize)
        self.size = (width,height)
        if os.path.exists(self.video_path) and os.path.splitext(self.video_path)[1] in VIDEO_EXTENSIONS:
            self.stream = cv2.VideoCapture(self.video_path)
            self.opened = self.stream.isOpened()
            if (not self.opened):
                raise RuntimeError('Error opening the video file')
            self.start()
        else:
            raise RuntimeError('Invalid video path')

