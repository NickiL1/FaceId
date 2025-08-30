import math
import tensorflow as tf
from itertools import product as product



class AnchorBoxes(object):
    def __init__(self, box_sizes, image_size, steps):
        self.box_sizes = box_sizes
        self.image_size = image_size
        self.steps = steps
        self.feature_maps = [[math.ceil(self.image_size[0] / step), math.ceil(self.image_size[1] / step)] for step in self.steps]

    def generate_anchors(self):
        anchors  = []

        for ind, map in enumerate(self.feature_maps):
            boxes = self.box_sizes[ind]
            for i, j in product(range(map[0]), range(map[1])):
                for box_size in boxes:
                    w,h = box_size, box_size
                    cx = (j + 0.5) * self.steps[ind]
                    cy = (i + 0.5) * self.steps[ind]
                    anchors.append([cx, cy, w, h])
        self.anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
        return self.anchors

    def normalize_anchors(self, boxes):
        div_tensor = tf.constant([self.image_size[0], self.image_size[1], self.image_size[0], self.image_size[1]], dtype=tf.float32)
        norm = tf.divide(boxes, div_tensor)
        return norm







def Iou(pivot, boxes):
    """
    expecting format of boxes to be [cx, cy, w, h] and tf tensors
    1 vs many iou
    """
    cx, cy, w, h = pivot
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2

    x1_ = boxes[:, 0] - boxes[:, 2]/2
    y1_ = boxes[:, 1] - boxes[:, 3]/2
    x2_ = boxes[:, 0] + boxes[:, 2]/2
    y2_ = boxes[:, 1] + boxes[:, 3]/2

    xA = tf.maximum(x1, x1_)
    yA = tf.maximum(y1, y1_)
    xB = tf.minimum(x2, x2_)
    yB = tf.minimum(y2, y2_)

    inter_area = tf.maximum(0.0, xB - xA) * tf.maximum(0.0, yB - yA)
    pivot_area = w * h
    boxes_area = boxes[:, 2] * boxes[:, 3]
    union_area = pivot_area + boxes_area - inter_area
    iou = inter_area / tf.maximum(union_area, 1e-8)
    return iou


def Iou_mat(boxes_1,boxes_2):
    """
    expecting format of boxes to be [cx, cy, w, h] and tf tensors
    many vs many iou
    returning a iou matrix
    """


    x1A, y1A = boxes_1[:, 0] - boxes_1[:, 2] / 2, boxes_1[:, 1] - boxes_1[:, 3] / 2
    x2A, y2A = boxes_1[:, 0] + boxes_1[:, 2] / 2, boxes_1[:, 1] + boxes_1[:, 3] / 2
    x1B, y1B = boxes_2[:, 0] - boxes_2[:, 2] / 2, boxes_2[:, 1] - boxes_2[:, 3] / 2
    x2B, y2B = boxes_2[:, 0] + boxes_2[:, 2] / 2, boxes_2[:, 1] + boxes_2[:, 3] / 2

    x1A = tf.expand_dims(x1A, 1)
    y1A = tf.expand_dims(y1A, 1)
    x2A = tf.expand_dims(x2A, 1)
    y2A = tf.expand_dims(y2A, 1)

    x1B = tf.expand_dims(x1B, 0)
    y1B = tf.expand_dims(y1B, 0)
    x2B = tf.expand_dims(x2B, 0)
    y2B = tf.expand_dims(y2B, 0)

    inter_x1 = tf.maximum(x1A, x1B)
    inter_y1 = tf.maximum(y1A, y1B)
    inter_x2 = tf.minimum(x2A, x2B)
    inter_y2 = tf.minimum(y2A, y2B)

    inter_w = tf.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = tf.maximum(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h

    areaA = (x2A - x1A) * (y2A - y1A)
    areaB = (x2B - x1B) * (y2B - y1B)

    union_area = areaA + areaB - inter_area

    iou = inter_area / tf.maximum(union_area, 1e-8)
    return iou








