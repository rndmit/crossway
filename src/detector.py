import os
import PIL
from numpy.core.defchararray import array
import yaml

import numpy as np
from sklearn.metrics import confusion_matrix
from skimage import measure
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from scipy.special import expit
from PIL import Image, ImageDraw
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def centroids_of_connected_components(bitmap, threshold=0.05, rescale=1.0):
    mask = bitmap > threshold
    bitmap = np.zeros_like(bitmap)
    bitmap[mask] = 1.0
    all_labels = measure.label(bitmap)
    centroids = []
    for region in measure.regionprops(label_image=all_labels):
        cx, cy = map(lambda p: int(p*rescale),
                     (region.centroid[0], region.centroid[1]))
        centroids.append((cx, cy))
    return centroids


def bboxes_of_connected_components(bitmap, threshold=0.05, rescale=1.0, area=1):
    mask = bitmap > threshold
    bitmap = np.zeros_like(bitmap)
    bitmap[mask] = 1.0
    all_labels = measure.label(bitmap, connectivity=None)
    bboxes = []
    for region in measure.regionprops(label_image=all_labels):
        # print(region.area, region.area < area, area)
        if region.area < area:
            continue
        x1, y1, x2, y2 = map(lambda p: int(
            p*rescale), (region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2]))
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def zero_centered_array_to_pil_image(orig_array):
    assert orig_array.dtype == np.float32
    h, w, c = orig_array.shape
    assert c == 3
    array = orig_array + 1  # 0.0 -> 2.0
    array *= 127.5  # 0.0 -> 255.0
    array = array.copy().astype(np.uint8)
    assert np.min(array) >= 0
    assert np.max(array) <= 255
    return Image.fromarray(array)


class Detector():

    zones = [
        [0, (180, 392), (269, 313), (372, 321), (288, 404)],
        [1, (817, 281), (1015, 286), (1026, 427), (818, 416)],
        [9, (372, 321), (315, 398), (808, 404), (812, 321)],
        [10, (744, 418), (990, 434), (1029, 720), (702, 720)],
        [7, (1132, 444), (1280, 436), (1280, 558), (1169, 583)]
    ]

    def __init__(self, width: int, height: int, base_filter_size: int, checkpoint_dir: str,
                 threshold=(0.2, 0.4)):
        self.model = None
        self.input_width = width
        self.input_height = height
        self.base_filter_size = base_filter_size
        self.checkpoint_dir = checkpoint_dir
        self.polygons = list(map(lambda p: Polygon(p[1:]), self.zones))

        self.human_threshold = threshold[0]
        self.car_threshold = threshold[1]

        self.construct()

    @staticmethod
    def get_zone_acc():
        return np.zeros((16))

    def in_zone(self, x1, y1, x2, y2):
        pgon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        for i, p in enumerate(self.polygons):
            if p.intersects(pgon):
                return self.zones[i][0]
        return -1

    @staticmethod
    def _conv_block(i, name, filters, strides):
        o = Conv2D(filters=filters, kernel_size=3,
                   strides=strides, padding='same')(i)
        o = BatchNormalization()(o)
        return ReLU()(o)

    def construct(self, use_skip_connections=True):
        # ----------------------
        inputs = Input(
            shape=(self.input_height, self.input_width, 3), name='inputs')
        e1 = Detector._conv_block(
            inputs, 'e1', filters=self.base_filter_size, strides=2)
        e2 = Detector._conv_block(e1, 'e2', filters=2 *
                                  self.base_filter_size, strides=2)
        e3 = Detector._conv_block(e2, 'e3', filters=4 *
                                  self.base_filter_size, strides=2)
        e4 = Detector._conv_block(e3, 'e4', filters=8 *
                                  self.base_filter_size, strides=2)

        d1 = UpSampling2D(name='e4nn')(e4)
        if use_skip_connections:
            d1 = Concatenate(name='d1_e3')([d1, e3])
        d1 = Detector._conv_block(d1, 'd1', filters=4 *
                                  self.base_filter_size, strides=1)

        d2 = UpSampling2D(name='d1nn')(d1)
        if use_skip_connections:
            d2 = Concatenate(name='d2_e2')([d2, e2])
        d2 = Detector._conv_block(d2, 'd2', filters=2 *
                                  self.base_filter_size, strides=1)

        d3 = UpSampling2D(name='d2nn')(d2)
        if use_skip_connections:
            d3 = Concatenate(name='d3_e1')([d3, e1])
        d3 = Detector._conv_block(
            d3, 'd3', filters=self.base_filter_size, strides=1)

        # d4 = UpSampling2D(name='d3nn')(d3)
        # d4 = conv_bn_relu_block(d3, 'd4', filters=base_filter_size, strides=1)

        logits = Conv2D(filters=2, kernel_size=1, strides=1,
                        activation=None, name='logits')(d3)

        self.model = Model(inputs=inputs, outputs=logits)
        # ----------------------

    def compile_model(self, learning_rate=0.001, pos_weight=1.0):

        def weighted_xent(y_true, y_predicted):
            return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_predicted, pos_weight=pos_weight))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate), loss=weighted_xent)

    def restore_model(self):
        checkpoint_info = yaml.load(
            open(os.path.join(self.checkpoint_dir, "checkpoint")).read())
        latest_ckpt = checkpoint_info['model_checkpoint_path']
        self.model.load_weights(os.path.join(self.checkpoint_dir, latest_ckpt))

    def predict(self, image: np.array):
        img = (image / 127.5) - 1.0     # нормализация палитры цветов
        prediction = expit(self.model.predict(np.expand_dims(img, 0))[0])

        # h, w, c = prediction.shape
        # rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
        # rgb_array[:, :, 0] = prediction[:, :, 0] * 255
        # rgb_array[:, :, 1] = prediction[:, :, 1] * 255

        h_centroids = centroids_of_connected_components(
            prediction[:, :, 0], threshold=self.human_threshold, rescale=2.0)
        # print("\t".join(map(str, [idx, filename])))
        debug_img = zero_centered_array_to_pil_image(img)
        canvas = ImageDraw.Draw(debug_img)
        for y, x in h_centroids:
            point = Point(x, y)
            for p in self.polygons:
                if p.contains(point):
                    canvas.rectangle((x-4, y-4, x+4, y+4), fill='blue')
                else:
                    canvas.rectangle((x-2, y-2, x+2, y+2), fill='red')
        c_centroids = centroids_of_connected_components(
            prediction[:, :, 1], threshold=self.car_threshold, rescale=2.0)
        for y, x in c_centroids:
            point = Point(x, y)
            for p in self.polygons:
                if p.contains(point):
                    canvas.rectangle((x-4, y-4, x+4, y+4), fill='blue')
                else:
                    canvas.rectangle((x-2, y-2, x+2, y+2), fill='yellow')
        for xy in self.zones:
            canvas.polygon(xy[1:], outline="wheat")

        return (np.array(debug_img), None)

    def predict_w_boxes(self, image: np.array):
        workload = Detector.get_zone_acc()

        img = (image / 127.5) - 1.0     # нормализация палитры цветов
        prediction = expit(self.model.predict(np.expand_dims(img, 0))[0])

        # Humans
        h_bboxes = bboxes_of_connected_components(
            prediction[:, :, 0], threshold=self.human_threshold, rescale=2.0, area=10)
        debug_img = zero_centered_array_to_pil_image(img)
        canvas = ImageDraw.Draw(debug_img)
        for x1, y1, x2, y2 in h_bboxes:
            z = self.in_zone(x1, y1, x2, y2)
            if z >= 0:
                canvas.rectangle((x1, y1, x2, y2), outline='blue')
                workload[z] += 1
            else:
                canvas.rectangle((x1, y1, x2, y2), outline='red')
        # Cars
        c_bboxes = bboxes_of_connected_components(
            prediction[:, :, 1], threshold=self.car_threshold, rescale=2.0, area=40)
        for x1, y1, x2, y2 in c_bboxes:
            z = self.in_zone(x1, y1, x2, y2)
            if z >= 0:
                canvas.rectangle((x1, y1, x2, y2), outline='blue')
                workload[z] += 1
            else:
                canvas.rectangle((x1, y1, x2, y2), outline='yellow')
        for xy in self.zones:
            canvas.polygon(xy[1:], outline="wheat")
        return (np.array(debug_img), workload)


if __name__ == '__main__':
    def parent_dir(path): return os.path.abspath(os.path.join(path, os.pardir))
    ckpt_dir = os.path.join(os.path.abspath(parent_dir(
        parent_dir(os.path.dirname(__file__)))), "data", "checkpoints")

    nn = Detector(1280, 720, 16, ckpt_dir)
    nn.compile_model()
    nn.restore_model()
    print(nn.model.summary())
