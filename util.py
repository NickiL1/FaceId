import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 32
def plot_image_with_bbox_batch(image, bbox):
    for image,box in zip(image,bbox):
        image = (image * 255).astype(np.uint8)
        image_bgr = image
        box = box.flatten().astype(np.int32)
        x1,y1,x2,y2,x3,y3,x4,y4 = box
        pts = np.array([(x1,y1), (x2,y2), (x3,y3), (x4,y4)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

        cv2.imshow("name", image_bgr)
        cv2.waitKey(0)