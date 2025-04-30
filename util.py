import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt

IMG_WIDTH = 1920
IMG_HEIGHT = 1080
def plot_image_with_bbox_batch(image, bbox):
    for image,box in zip(image,bbox):
        image = np.array(image, copy=True)
        image = (image * 255).astype(np.uint8)
        cv2.rectangle(image,
                      tuple(np.multiply(box[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(box[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)
        cv2.imshow("sample",image)
        cv2.waitKey(0)
