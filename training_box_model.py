import os.path
import uuid
import keras
from PIL import Image
import numpy as np
import random
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from util import plot_image_with_bbox_batch
from shapely.geometry import Polygon


import pandas as pd
"""
these shapes are the celebA dataset standard. 
"""
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 64

images_folder = "../Training Data/archive (2)/img_align_celeba/img_align_celeba/"
augmented_images_folder = '../Training Data/augmented_images/'
box_file = "../Training Data/archive (2)/list_bbox_celeba.csv"
augmented_file = '../Training Data/augmented.csv'

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
def parse_line_normal(line):
    parts = tf.strings.split(line,sep=",")
    image_id = parts[0]
    width = tf.strings.to_number(parts[3], tf.int32)
    height = tf.strings.to_number(parts[4], tf.int32)
    x1 = tf.strings.to_number(parts[1], tf.int32)
    y1 = tf.strings.to_number(parts[2], tf.int32)
    x2, y2 = x1 + width, y1
    x3,y3 = x1 + width,y1 + height
    x4,y4 = x1, y1 + height
    box = tf.stack([x1,y1,x2,y2,x3,y3,x4,y4])
    return  image_id, box
def parse_line_augmented(line):
    parts = tf.strings.split(line,sep=",")
    image_id = parts[0]
    x1 = tf.strings.to_number(parts[1], tf.int32)
    y1 = tf.strings.to_number(parts[2], tf.int32)
    x2 = tf.strings.to_number(parts[3], tf.int32)
    y2 = tf.strings.to_number(parts[4], tf.int32)
    x3 = tf.strings.to_number(parts[5], tf.int32)
    y3 = tf.strings.to_number(parts[6], tf.int32)
    x4 = tf.strings.to_number(parts[7], tf.int32)
    y4 = tf.strings.to_number(parts[8], tf.int32)

    box = tf.stack([x1,y1,x2,y2,x3,y3,x4,y4])
    return  image_id, box

def get_image_normal(image_id, box):
    image_path = tf.strings.join([images_folder,image_id])
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, (IMG_HEIGHT,IMG_WIDTH))
    image = tf.cast(image,tf.float32) / 255.
    image = tf.reverse(image, axis=[-1])
    return image,box
def get_image_augmented(image_id, box):
    image_path = tf.strings.join([augmented_images_folder,image_id])
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, (IMG_HEIGHT,IMG_WIDTH))
    image = tf.cast(image,tf.float32) / 255.
    image = tf.reverse(image, axis=[-1])
    return image,box
def filter_overflow(image,box):
    x1,y1,x2,y2,x3,y3,x4,y4 = tf.unstack(box)
    # Valid if bbox is inside the image and positive
    valid1 = tf.logical_and(
        tf.logical_and(
            tf.logical_and(x1 >= 0, y1 > 0),
            tf.logical_and(x1 <= IMG_WIDTH, y1 <= IMG_HEIGHT)
        ),
        tf.logical_and(
            tf.logical_and(x2 >= 0, y2 > 0),
            tf.logical_and(x2 <= IMG_WIDTH, y2 <= IMG_HEIGHT)
        )
    )
    valid2 = tf.logical_and(
        tf.logical_and(
            tf.logical_and(x3 >= 0, y3 > 0),
            tf.logical_and(x3 <= IMG_WIDTH, y3 <= IMG_HEIGHT)
        ),
        tf.logical_and(
            tf.logical_and(x4 >= 0, y4 > 0),
            tf.logical_and(x4 <= IMG_WIDTH, y4 <= IMG_HEIGHT)
        )
    )
    valid = tf.logical_and(valid1,valid2)
    return valid

def augmentation(dataset, ignore=1):
    if ignore == 1: return
    aug_dir = '../Training Data/augmented_images'
    for filename in os.listdir(aug_dir):
        file_path = os.path.join(aug_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    aug_dir = '../Training Data/augmented_images/'
    box_coords = []
    image_ids =[]
    columns = ['image_id', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    for image,box in dataset:
        image_np = image.numpy()
        box_np = box.numpy()
        x1,y1,x2,y2,x3,y3,x4,y4 = box_np
        for _ in range(2):
            center = (IMG_WIDTH // 2, IMG_HEIGHT // 2)
            angle = np.random.randint(-360, 360)
            scale = np.random.uniform(0.5,1)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated_scaled_image = cv2.warpAffine(image_np, M, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32)
            points = np.array([
                [x1,y1,1],
                [x2,y2,1],
                [x3,y3,1],
                [x4,y4,1]
            ])
            new_box = M.dot(points.T).T.astype(np.int32)
            new_box = new_box.flatten()
            box_coords.append(new_box)

            image_id = str(uuid.uuid4()) + ".jpg"
            image_path = aug_dir + image_id
            image_bgr = (rotated_scaled_image * 255).astype(np.uint8)
            # image_bgr = cv2.cvtColor(image_bgr,cv2.COLOR_RGB2BGR)
            # new_box = new_box.reshape((-1, 1, 2))
            # cv2.polylines(image_bgr, [new_box], isClosed=True, color=(0, 255, 0), thickness=1)
            # cv2.imshow("dd", image_bgr)
            # cv2.waitKey(0)
            cv2.imwrite(image_path,image_bgr)
            image_ids.append(image_id)


    df = pd.DataFrame(box_coords, columns=columns[1:])
    df['image_id'] = image_ids
    df = df[['image_id'] + columns[1:]]
    df.to_csv('../Training Data/augmented.csv',index=False)


dataset_normal = tf.data.TextLineDataset(box_file).skip(2)
dataset_normal = dataset_normal.map(parse_line_normal,num_parallel_calls=tf.data.AUTOTUNE)
dataset_normal = dataset_normal.map(get_image_normal, num_parallel_calls=tf.data.AUTOTUNE)
dataset_normal = dataset_normal.filter(filter_overflow)
dataset_normal = dataset_normal.shuffle(4_000).take(3_000)
augmentation(dataset_normal, ignore=1)

dataset_augmented = tf.data.TextLineDataset(augmented_file).skip(2)
dataset_augmented = dataset_augmented.map(parse_line_augmented,num_parallel_calls=tf.data.AUTOTUNE)
dataset_augmented = dataset_augmented.map(get_image_augmented, num_parallel_calls=tf.data.AUTOTUNE)
dataset_augmented = dataset_augmented.shuffle(6_000)

final_dataset = dataset_normal.concatenate(dataset_augmented)
final_dataset = final_dataset.shuffle(9_000)

train_dataset = final_dataset.take(7_200).batch(BATCH_SIZE)
devtest_dataset = final_dataset.skip(7_200)
test_dataset = devtest_dataset.take(900).batch(BATCH_SIZE)
val_dataset = devtest_dataset.skip(900).batch(BATCH_SIZE)

# sample = val_dataset.as_numpy_iterator().next()
# image,box = sample
# plot_image_with_bbox_batch(image,box)

def build_model():
    inputs = tf.keras.Input((IMG_HEIGHT,IMG_WIDTH,3))

    # Feature extractor with more layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)  # Batch Normalization
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)  # Batch Normalization
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)  # Batch Normalization
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)  # Batch Normalization
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)  # Batch Normalization
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout for regularization
    x = layers.Dense(512, activation='relu')(x)  # A smaller dense layer
    x = layers.Dropout(0.4)(x)  # Additional dropout to prevent overfitting

    # Output: 8 numbers (x1, y1, x2, y2, x3, y3, x4, y4)
    outputs = layers.Dense(8)(x)
    model = tf.keras.Model(inputs, outputs)
    return model


# class PolygonIoULoss(tf.keras.losses.Loss):
#     def __init__(self):
#         super().__init__()
#
#     def mse_loss(self, y_true, y_pred):
#         return tf.reduce_mean(tf.square(y_true - y_pred))
#
#     def polygon_iou_loss(self, y_true, y_pred):
#         # y_true and y_pred shape: (batch_size, 8)
#         @tf.py_function(Tout=tf.float32)
#         def single_iou(sample_tuple):
#             y_true, y_pred = sample_tuple
#             print(type(y_true))
#             y_true_np = y_true.numpy()
#             y_pred_np = y_pred.numpy()
#             y_true_np = y_true_np.reshape(4, 2)
#             y_pred_np = y_pred_np.reshape(4, 2)
#
#             true_poly = Polygon(y_true_np)
#             pred_poly = Polygon(y_pred_np)
#
#             if not true_poly.is_valid or not pred_poly.is_valid:
#                 return np.float32(1.0)
#
#             intersection = true_poly.intersection(pred_poly).area
#             union = true_poly.union(pred_poly).area
#             iou = intersection / union if union > 0 else 0.0
#             return tf.stop_gradient(np.float32(1.0 - iou))
#
#         # Map single_iou over batch using tf.py_function
#         batch_iou_losses = tf.map_fn(
#             single_iou,
#             (y_true,y_pred),
#             fn_output_signature=tf.float32
#         )
#         return tf.reduce_mean(batch_iou_losses)
#
#     def call(self, y_true, y_pred):
#         mse = self.mse_loss(y_true, y_pred)
#         poly = self.polygon_iou_loss(y_true, y_pred)
#         return mse + poly
#



# Create the model
model = build_model()

model.compile(
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5),
    loss="mse",  # mean squared error, good for coordinate regression
)

logdir = "logs/fit"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(train_dataset,validation_data=val_dataset, epochs=75,callbacks=[tensorboard_callback])

print(model.evaluate(test_dataset))
model.save("face_detect_model.h5")



