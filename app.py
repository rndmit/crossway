import os
import sys
import time
from typing import Any, Union, Optional
from threading import Thread
from PIL.Image import Image

import numpy as np
import cv2

from config import AppConfig, NNConfig
from src.capture import AsyncCapture
from src.detector import Detector
from src.mqtt import MQTTClient



class App(object):
    def __init__(self):
        # Capture
        print('Loading stream')
        self.stream_capture = AsyncCapture(AppConfig.stream_url, 1/12, 2)

        # Detector
        print('Compiling and loading model')
        self.detector = Detector(1280, 720, 16, os.path.join(
            os.path.dirname(__file__), "data", "checkpoints"),
            (NNConfig.human_threshold, NNConfig.car_threshold))
        self.detector.compile_model()
        self.detector.restore_model()
        print(self.detector.model.summary())

        # MQTT
        print('Loading MQTT client')
        self.mqtt = MQTTClient(AppConfig.mqtt_username)
        self.mqtt.connect()
        print('Ready')


if __name__ == '__main__':
    app = App()
    while True:
        frame = app.stream_capture.get_frame()
        if frame is not None:
            frame = np.array(cv2.resize(frame, (1280, 720)), dtype=np.float32)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result, workload = app.detector.predict_w_boxes(frame)
            #app.mqtt.send_workload(workload)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imshow('Output', result)
        key = cv2.waitKey(app.stream_capture.FPS_MS)
        if AppConfig.tweak_mode:
            if key == ord('a'):
                app.detector.human_threshold += AppConfig.tm_threshold_step
            elif key == ord('z'):
                app.detector.human_threshold -= AppConfig.tm_threshold_step
            elif key == ord('s'):
                app.detector.car_threshold += AppConfig.tm_threshold_step
            elif key == ord('x'):
                app.detector.car_threshold -= AppConfig.tm_threshold_step
            sys.stdout.write("\r" + "H threshold:" + str(app.detector.human_threshold) + " | "
                             "C threshold:" + str(app.detector.car_threshold)
                             )  # sorry about this
