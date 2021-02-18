import time
import random
import numpy as np
import client
import threading

CARSPEED = 2
WALKERSPEED = 5
WIDTH = 5
CHANGETIME = 4
TACT = 10

idx = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
]

sat = [[1, 10], [1, 4], [7, 10], [7, 4], [0, 9], [0, 3], [0, 6], [9, 3], [3, 7], [6, 9], [0, 10], [0, 7], [
    9, 7], [9, 4], [6, 4], [6, 1], [3, 1], [3, 10], [12, 1], [13, 10], [14, 11], [14, 7], [15, 0], [15, 4]]

for i in range(5, 10):
    sat.append([12, i])
for i in range(2, 7):
    sat.append([13, i])
for i in range(0, 4):
    sat.append([14, i])
for i in range(8, 12):
    sat.append([15, i])


def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t


class Mask():
    def __init__(self, sat, mask=None):
        self.sat = sat
        self.mask = [] if mask is None else mask

    @staticmethod
    def Generate(sat):
        mask = Mask(sat)
        while True:
            for _ in range(16):
                mask.append(random.randint(0, 1))
            if mask.is_possible():
                return mask

    @staticmethod
    def FromNumber(sat, a):
        m = Mask(sat)
        for _ in range(16):
            m.append(a % 2)
            a //= 2
        return m

    def no_block(self, blockzone):
        blockmask = np.zeros((12))
        for j in range(0, 4):
            if(blockzone[j]):
                for i in range(j*3, 3+j*3):
                    blockmask[i] = 1
                for i in range(j*3+3, j*3+11, 4):
                    blockmask[i % 12] = 1

        for i in range(12):
            if(self.mask[i] and blockmask[i]):
                return 0
        return 1

    def append(self, obj):
        self.mask.append(obj)

    def is_possible(self):
        for i in self.sat:
            if not ((self.mask[i[0]] == 0) or (self.mask[i[1]] == 0)):
                # print("is not possible sat", self.mask)
                return False
        for i in idx:
            if not (self.mask[i[0]] == self.mask[i[1]] and self.mask[i[1]] == self.mask[i[2]]):
                # print("is not possible", self.mask)
                return False
        return True

    def __getitem__(self, key):
        return self.mask[key]


class Control():
    def __init__(self, sat, cl):
        self.cl = cl
        self.sat = sat
        self.last_run = 0
        self.prevmask = Mask(sat, np.zeros((16)))
        self.__init__attrs()
        self.zone = np.zeros((16))
        self.workload = np.zeros((16))

    def __init__attrs(self):
        self.contflow = []
        self.changeflow = []
        self.weights = []
        self.timeweights = []
        self.times = []

        for i in range(16):
            self.contflow.append(
                TACT//(CARSPEED*(i < 12)+(i >= 12)*WALKERSPEED))
            self.changeflow.append((TACT-CHANGETIME)//(CARSPEED *
                                                       (i < 12)+(i >= 12)*WALKERSPEED))
            self.weights.append(1)
            self.timeweights.append(1)
            self.times.append(1)

            if i < 12:
                self.timeweights[i] /= 2
            if i >= 12:
                self.contflow[i] *= WIDTH
                self.changeflow[i] *= WIDTH
                self.weights[i] /= 3

    def __greed(self, cur_mask, zone):
        ans = 0
        for i in range(16):
            if(self.prevmask[i] == cur_mask[i]):
                ans += min(zone[i], self.contflow[i])*cur_mask[i] * \
                    self.weights[i] * \
                    (((self.times[i]*self.timeweights[i])/30)**2)
            else:
                ans += min(zone[i], self.changeflow[i])*cur_mask[i] * \
                    self.weights[i] * \
                    (((self.times[i]*self.timeweights[i])/30)**2)
        return ans

    def save(self, zone, workload):
        print('save')
        self.zone = zone
        self.workload = workload

    def start_loop(self):
        def check():
            print('time', time.time() - self.last_run)
            if time.time() - self.last_run < TACT:
                return
            print('calc mask')
            mask, val = control.greedchoose(self.zone, self.workload)
            self.cl.send_control_signal(mask, val)

        set_interval(check, 1)

    def greedchoose(self, zone, workload, masksize=16):
        maxv = 0
        maxmask = []
        prevmask = self.prevmask
        blockzone = np.zeros((4))
        blockzone[:4] = workload[8:12]
        for i in range(1 << masksize):
            mask = Mask.FromNumber(self.sat, i)

            if mask.is_possible():
                a = self.__greed(mask, zone)
                if(maxv <= a):
                    maxv = a
                    maxmask = mask

        for i in range(16):
            if (maxmask[i]):
                self.times[i] = 1
                if(prevmask[i] == maxmask[i]):
                    zone[i] -= min(zone[i], self.contflow[i])
                else:
                    zone[i] -= min(zone[i], self.changeflow[i])
            elif (zone[i]):
                self.times[i] += TACT

        self.prevmask = maxmask
        self.last_run = time.time()

        return [maxmask, maxv]


if __name__ == "__main__":
    cl = client.MQTTClient(client.mqtt_username)
    cl.connect()
    control = Control(sat, cl)

    def handler(payload):
        workload = payload['context']['workload']
        workload[6:9] = workload[7:10]
        workload[10] = 0
        buf = np.zeros((16))
        buf[12:16] = workload[:4]
        buf[:3] = workload[7]
        buf[3:6] = workload[6]
        buf[6:9] = workload[5]
        buf[9:12] = workload[4]
        print('new workload')
        control.save(buf, workload)

    cl.subscribe_to_workload(handler)
    control.start_loop()
    cl.loop()
