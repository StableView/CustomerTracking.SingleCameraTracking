from src.video.stream import VideoStream
from threading import Thread
from queue import Queue
import cv2
import os

class VideoThread(VideoStream): 
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

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                print("Thread stopped")
                return
            
            if not self.Q.full():

                (self.grabbed, self.frame) = self.stream.read()
                if not self.grabbed:
                    self.stop()
                    return
                if self.size[0] != None and self.size[1] != None:
                    self.frame = cv2.resize(self.frame, self.size, interpolation=cv2.INTER_LINEAR)
                self.Q.put(self.frame)
        
    def read(self):
        return self.Q.get(timeout=2)
    
    def is_opened(self):
        self.opened = self.stream.isOpened()
        if not self.opened:
            self.stop()
        return self.opened
    
    def stop(self):
        self.stopped = True
        self.stream.release()
        print('Stop Stream')


