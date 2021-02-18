import paho.mqtt.client as mqtt
import json
import numpy as np

mqtt_username = "13fa60a0f3e5d2416b4ede1ea38a33684117f57325b38bc57476bc4b7c2c"


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
            'context': wl
        })
        self.client.publish('/devices/crossway/workload', payload)

    def send_control_signal(self, mask, val):
        payload = json.dumps({
            'bl': {
                'value': mask[0]
            },
            'br': {
                'value': mask[1]
            },
            'bs': {
                'value': mask[2]
            },
            'rl': {
                'value': mask[3]
            },
            'rr': {
                'value': mask[4]
            },
            'rs': {
                'value': mask[5]
            },
            'tl': {
                'value': mask[6]
            },
            'tr': {
                'value': mask[7]
            },
            'ts': {
                'value': mask[8]
            },
            'll': {
                'value': mask[9]
            },
            'lr': {
                'value': mask[10]
            },
            'ls': {
                'value': mask[11]
            },
            'tc': {
                'value': mask[12]
            },
            'rc': {
                'value': mask[13]
            },
            'bc': {
                'value': mask[14]
            },
            'lc': {
                'value': mask[15]
            },
            'val': {
                'value': val
            },
        })
        print('payload', payload)
        self.client.publish('/devices/crossway', payload)

    def subscribe_to_workload(self, cb):
        self.client.subscribe('/devices/crossway/workload')

        def on_message(client, userdata, msg):
            payload = json.loads(msg.payload)
            cb(payload)

        self.client.on_message = on_message

    def loop(self):
        self.client.loop_forever()
