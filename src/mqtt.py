import paho.mqtt.client as mqtt
import json
import numpy as np
from threading import Thread, ThreadError

class MQTTClient():
    def __init__(self, username):
        self.username = username
        self.client = mqtt.Client()

    # def __on_connect():
    #     print("Connected to greenpl mqtt broker")

    def connect(self):
        self.client.username_pw_set(self.username, '1')
        self.client.connect("mqtt.greenpl.ru", 1883)

    def send_workload(self, wl):
        payload = json.dumps({
            'value': np.sum(wl),
            'context': {
                'workload': list(wl)
            },
        })
        self.client.publish('/devices/crossway/workload', payload)

    def subscribe_to_workload(self, cb):
        self.client.subscribe('/devices/crossway/workload')

        def on_message(client, userdata, msg):
            payload = json.loads(msg.payload)
            cb(payload)

        self.client.on_message = on_message
