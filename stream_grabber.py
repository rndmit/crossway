import os
import sys
import time
import datetime
from threading import Thread, ThreadError

import cv2
import numpy as np

from src.stream.capture import AsyncCapture

if __name__=='__main__':
    dst = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'imgs')
    delay = int(sys.argv[1])
    cap = AsyncCapture('https://sochi.camera:8081/cam_274/tracks-v1/mono.m3u8')
    while True:
        frame = cap.get_frame()
        if frame is not None:
            filepath = os.path.join(dst, f'cap-{datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}.png')
            print(f'Grabbed: {filepath}')
            cv2.imwrite(filepath, frame)
            cv2.imshow('frame', frame)
        cv2.waitKey(cap.FPS_MS)
        time.sleep(delay)