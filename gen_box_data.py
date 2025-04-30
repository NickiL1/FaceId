import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
import os
import uuid
import shutil
import time
import json
from util import plot_image_with_bbox_batch
from sklearn.model_selection import train_test_split
import albumentations as alb

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

IMG_WIDTH = 1920
IMG_HEIGHT = 1080
BATCH_SIZE = 32
tf.random.set_seed(42)
np.random.seed(42)

def collect_images(n_images=30, num_of_collections=1,new=True):
    images_path = "data/images"
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    else:
        if new:
            for filename in os.listdir(images_path):
                file_path = os.path.join(images_path,filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print("removed file: " + file_path)
    cap = cv2.VideoCapture(0)
    for iteration in range(num_of_collections):
        for imgnum in range(n_images):
            print('Collecting image {}'.format(imgnum))
            _, frame = cap.read()
            print(frame.shape)
            image_name = os.path.join(images_path,f'{str(uuid.uuid1())}.jpg')
            cv2.imwrite(image_name,frame)
            cv2.imshow("frame",frame)
            time.sleep(0.5)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("press any key to continue for next iteration")
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

# collect_images(30,4,new=True) to collect new images uncomment. to add images without deleting existing set new=False


def get_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img,(720,720))
    bgr = tf.reverse(img, axis=[-1])
    return bgr


def create_test_val_train_folders(images_list ,train_test_split_ratio=0.2,test_val_split=0.5):
    path_list = []
    for path in images_list:
        path_list.append(path.numpy().decode("utf-8"))
    train_paths, test_paths = train_test_split(path_list,test_size=train_test_split_ratio,random_state=42)
    test_paths, val_paths = train_test_split(test_paths,test_size=test_val_split,random_state=42)

    for folder in ["train", "test", "val"]:
        if not os.path.exists(f"data/{folder}/images"):
            os.makedirs(f"data/{folder}/images")
        else:
            for filename in os.listdir(f"data/{folder}/images"):
                full_path = os.path.join(f"data/{folder}/images",filename)
                os.remove(full_path)
                print("removed file: " + full_path)
        if not os.path.exists(f"data/{folder}/labels"):
            os.makedirs(f"data/{folder}/labels")
        else:
            for filename in os.listdir(f"data/{folder}/labels"):
                full_path = os.path.join(f"data/{folder}/labels",filename)
                os.remove(full_path)
                print("removed file: " + full_path)
        if folder == "train":
            paths = train_paths
        elif folder == "test":
            paths = test_paths
        else:
            paths = val_paths
        for path in paths:
            print(f"moving {path} to data/{folder}/images")
            shutil.copy(path,f"data/{folder}/images")
            label_filename = path.split("/")[3].split(".")[0] + ".json"
            label_path = os.path.join("data/labels",label_filename)
            if os.path.exists(label_path):
                print(f"moving {label_path} to data/{folder}/labels")
                shutil.copy(label_path,f"data/{folder}/labels")


def augment_partition(partition, augmentor):
    for image in os.listdir(os.path.join('data', partition, 'images')):
        img = cv2.imread(os.path.join('data', partition, 'images', image))

        coords = [0, 0, 0.00001, 0.00001]
        label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT]))

        try:
            for x in range(60):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'),
                            augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0, 0, 0, 0]
                    annotation['class'] = 0

                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'),
                          'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)


def create_aug_set():

    if not os.path.exists("aug_data"):
        os.makedirs("aug_data")
    augmentor = alb.Compose([alb.RandomCrop(width=720, height=720),
                             alb.HorizontalFlip(p=0.5),
                             alb.RandomBrightnessContrast(p=0.2),
                             alb.RandomGamma(p=0.2),
                             alb.RGBShift(p=0.2),
                             alb.VerticalFlip(p=0.5)],
                            bbox_params=alb.BboxParams(format='albumentations',
                                                       label_fields=['class_labels']))
    for partition in ['train', 'test', 'val']:
        if not os.path.exists(f"aug_data/{partition}/images"):
            os.makedirs(f"aug_data/{partition}/images")
        else:
            for filename in os.listdir(f"aug_data/{partition}/images"):
                full_path = os.path.join(f"aug_data/{partition}/images",filename)
                os.remove(full_path)
                print("removed file: " + full_path)
        if not os.path.exists(f"aug_data/{partition}/labels"):
            os.makedirs(f"aug_data/{partition}/labels")
        else:
            for filename in os.listdir(f"aug_data/{partition}/labels"):
                full_path = os.path.join(f"aug_data/{partition}/labels",filename)
                os.remove(full_path)
                print("removed file: " + full_path)
        augment_partition(partition,augmentor)

def parse_label(path):
    with open(path.numpy(),"r",encoding="utf-8") as file:
        label = json.load(file)
    return [label["class"]], label["bbox"]
def get_labels(path):
    cls, bbox = tf.py_function(parse_label, [path], [tf.uint8, tf.float16])
    cls.set_shape([1])
    bbox.set_shape([4])
    return (cls, bbox)
# create_test_val_train_folders(images_list)
# create_aug_set()


train_images = tf.data.Dataset.list_files("aug_data/train/images/*.jpg", shuffle=False)
train_images = train_images.map(get_image)
train_images = train_images.map(lambda x: tf.image.resize(x,(120,120)))
train_images = train_images.map(lambda x: x / 255)

test_images = tf.data.Dataset.list_files("aug_data/test/images/*.jpg", shuffle=False)
test_images = test_images.map(get_image)
test_images = test_images.map(lambda x: tf.image.resize(x,(120,120)))
test_images = test_images.map(lambda x: x / 255)

val_images = tf.data.Dataset.list_files("aug_data/val/images/*.jpg", shuffle=False)
val_images = val_images.map(get_image)
val_images = val_images.map(lambda x: tf.image.resize(x,(120,120)))
val_images = val_images.map(lambda x: x / 255)

train_labels = tf.data.Dataset.list_files("aug_data/train/labels/*.json",shuffle=False)
train_labels = train_labels.map(get_labels)


test_labels = tf.data.Dataset.list_files("aug_data/test/labels/*.json", shuffle=False)
test_labels = test_labels.map(get_labels)

val_labels = tf.data.Dataset.list_files("aug_data/val/labels/*.json", shuffle=False)
val_labels = val_labels.map(get_labels)

train_dataset = tf.data.Dataset.zip((train_images,train_labels))
train_dataset = train_dataset.shuffle(6_000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.zip((test_images,test_labels))
test_dataset = test_dataset.shuffle(1_000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.zip((val_images,val_labels))
val_dataset = val_dataset.shuffle(1_000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# sample = train_dataset.as_numpy_iterator().next()
# image,box = sample
# plot_image_with_bbox_batch(image,box[1])

def localization_loss(y_true, y_pred):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - y_pred[:, :2]))

    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]

    h_pred = y_pred[:, 3] - y_pred[:, 1]
    w_pred = y_pred[:, 2] - y_pred[:, 0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    return delta_coord + delta_size


def build_my_model():
    inputs = tf.keras.Input((120, 120, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    # classification:
    c1 = layers.Dense(2048, activation="relu")(x)
    c2 = layers.Dense(1, activation="sigmoid")(c1)

    # bounding box coordinate regression:
    r1 = layers.Dense(2048, activation="relu")(x)
    r2 = layers.Dense(4, activation="sigmoid")(r1)  # coordinates are normalized
    model = tf.keras.Model(inputs=inputs, outputs=[c2, r2])
    return model

def build_pre_trained_model():
    input_layer = layers.Input(shape=(120, 120, 3))

    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model
    f1 = layers.GlobalMaxPooling2D()(vgg)
    class1 = layers.Dense(2048, activation='relu')(f1)
    class2 = layers.Dense(1, activation='sigmoid')(class1)

    # Bounding box model
    f2 = layers.GlobalMaxPooling2D()(vgg)
    regress1 = layers.Dense(2048, activation='relu')(f2)
    regress2 = layers.Dense(4, activation='sigmoid')(regress1)

    facetracker = tf.keras.Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker


class FaceDetection(tf.keras.Model):

    def __init__(self,base_model, **kwargs):
        super().__init__(**kwargs)
        self.model = base_model

    def compile(self, class_loss, localization_loss, optimizer, **kwargs):
        super().compile(**kwargs)
        self.closs = class_loss
        self.lloss = localization_loss

    def train_step(self, batch, **kwargs):
        X, y = batch
        cls, bbox = y
        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(cls, classes)
            batch_localizationloss = self.lloss(tf.cast(bbox, tf.float32), coords)

            total_loss = batch_localizationloss + batch_classloss

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)
        cls, bbox = y
        batch_classloss = self.closs(cls, classes)
        batch_localizationloss = self.lloss(tf.cast(bbox, tf.float32), coords)
        total_loss = batch_localizationloss + batch_classloss

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)

logdir = "logs/fit_my"
tensorboard_cll_my = tf.keras.callbacks.TensorBoard(log_dir=logdir)
model = FaceDetection(build_my_model())
model.compile(
    class_loss=tf.keras.losses.BinaryCrossentropy(),
    localization_loss=localization_loss,
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5)
)

model.summary()
history = model.fit(train_dataset,epochs=70,validation_data=val_dataset,callbacks=[tensorboard_cll_my])
model.save("my_model.h5")
print(model.evaluate(test_dataset))


logdir = "logs/fit_pre"
tensorboard_cll_pre = tf.keras.callbacks.TensorBoard(log_dir=logdir)
pre_trained = FaceDetection(build_pre_trained_model())
pre_trained.compile(
    class_loss=tf.keras.losses.BinaryCrossentropy(),
    localization_loss=localization_loss,
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5)
)
pre_trained.summary()
history_pre = pre_trained.fit(train_dataset,epochs=70,validation_data=val_dataset,callbacks=[tensorboard_cll_pre])
pre_trained.save("pre_trained_model.h5")
print(pre_trained.evaluate(test_dataset))
