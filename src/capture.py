import os
import time
from threading import Thread, ThreadError

import cv2

class AsyncCapture(object):
    frame = None 
    status = None

    def __init__(self, src=0, output_fps=1/30,
        buffer_size=1):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        self.FPS = output_fps
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def get_frame(self):
        if self.status:
            return self.frame