import keras.src.saving
import tensorflow as tf
import os
import cv2
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, BatchNormalization, PReLU, DepthwiseConv2D, Add, Lambda, Flatten, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy, Loss, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
import math
import random
import numpy as np


INPUT_W = 112
INPUT_H = 112
INPUT_CHANNELS = 3
dataset_path = "../ms1m-arcface/faces_dataset.tfrecord"
debug_dataset = "../ms1m-arcface/debug_dataset.tfrecord"
tf.random.set_seed(42)
random.seed(42)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_shuffled_tfrecord(data_dir, output_path, debug, shuffle):
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    int_class_names = sorted([int(s) for s in class_names])
    if debug:
        int_class_names = int_class_names[:5]
    class_names = [str(s) for s in int_class_names]
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    files_list = []
    for class_name, class_idx in class_to_index.items():
        class_dir = os.path.join(data_dir, class_name)
        print(f"working on folder: {class_dir}")
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                image_path = os.path.join(class_dir, filename)
                files_list.append((image_path, class_idx))
    print(len(files_list))
    if shuffle:
        print("shuffling")
        random.shuffle(files_list)

    with tf.io.TFRecordWriter(output_path) as writer:
        for image_path, label in files_list:
            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()

            example = tf.train.Example(features=tf.train.Features(feature={
                        'image_raw': _bytes_feature(image_bytes),
                        'label': _int64_feature(label)
                    }))
            print(f"added image {image_path} to dataset with label {label}")
            writer.write(example.SerializeToString())
    print(f"TFRecord saved to {output_path}")



def parse_sample(example_proto):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(parsed_features['image_raw'], channels=INPUT_CHANNELS)
    label = parsed_features['label']
    label = tf.cast(label, tf.int32)
    return image, label

def preprocess_image(image, label):
    image = tf.image.resize(image, (INPUT_W, INPUT_H))
    image = tf.subtract(image, 127.5)
    image = tf.multiply(image, 0.0078125)
    return image, label

def load_dataset(tfrecord_path, batch_size=None,shuffle_buffer=None):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_sample)
    dataset = dataset.map(preprocess_image)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer)
    return dataset


class ArcFace(Layer):
    def __init__(self, n_classes=85_742, s=64.0, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s

    def build(self, input_shape):
        self.m = self.add_weight(name='m', shape=(), initializer="zeros", trainable=False, dtype=tf.float32)

        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=l2(4e-5))
        super(ArcFace, self).build(input_shape[0])

    def call(self, inputs):
        pi = tf.constant(math.pi, dtype=tf.float32)
        cos_m = tf.cos(self.m)
        sin_m = tf.sin(self.m)
        mm = sin_m * self.m
        threshold = tf.cos(pi - self.m)


        embedding, labels = inputs
        embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
        embedding = tf.divide(embedding, embedding_norm, name='norm_embedding')
        weights = self.W
        weights_norm = tf.norm(weights, axis=0, keepdims=True)
        weights = tf.divide(weights, weights_norm, name='norm_weights')
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = self.s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = self.s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = labels
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(self.s, cos_t, name='scalar_cos_t')

        logits = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
        return logits

    def compute_output_shape(self, input_shape):
        return None, self.n_classes

    def get_config(self):
        base_config = super().get_config()
        return {**base_config ,'s': self.s, 'n_classes': self.n_classes, "m": self.m.numpy()}


@keras.src.saving.register_keras_serializable(package='ArcFace')
class Bottleneck(Model):
    def __init__(self, filters, t, strides, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.filters = filters
        self.t = t
        self.strides = strides

        self.conv1x1_1 = Conv2D(filters * t, 1, padding='same', use_bias=False, kernel_regularizer=l2(4e-5), kernel_initializer="he_normal")
        self.bn1 = BatchNormalization()
        self.prelu1 =  PReLU(shared_axes=[1,2])

        self.dwise3x3 = DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False, depthwise_regularizer=l2(4e-5),depthwise_initializer="he_normal")
        self.bn2 = BatchNormalization()
        self.prelu2 = PReLU(shared_axes=[1,2])

        self.conv1x1_2 = Conv2D(filters, 1, padding='same', use_bias=False, kernel_regularizer=l2(4e-5), kernel_initializer="he_normal")
        self.bn3 = BatchNormalization()
        self.add = Add()

    def build(self, input_shape):
        super(Bottleneck, self).build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1x1_1(inputs)
        x = self.bn1(x,training=training)
        x = self.prelu1(x)

        x = self.dwise3x3(x)
        x = self.bn2(x,training=training)
        x = self.prelu2(x)

        x = self.conv1x1_2(x)
        x = self.bn3(x,training=training)

        if self.strides == 1:
            x =self.add([x, inputs])
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'strides': self.strides, 'filters': self.filters, 't': self.t}


@keras.src.saving.register_keras_serializable(package='ArcFace')
class BottleneckBlock(Model):
    def __init__(self, filters, t, strides, n, **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)
        self.filters = filters
        self.t = t
        self.strides = strides
        self.n = n

        self.bottleneck_list = []
        if strides == 2:
            self.bottleneck_list.append(Bottleneck(filters=filters, t=t, strides=strides))
            n = n - 1
        for i in range(n):
            self.bottleneck_list.append(Bottleneck(filters=filters, t=t, strides=1))

    def build(self, input_shape):
        super(BottleneckBlock, self).build(input_shape)

    def call(self, inputs, training=False):
        x = inputs
        for bottleneck in self.bottleneck_list:
            x = bottleneck(x, training=training)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'strides': self.strides, 'filters': self.filters, 't': self.t, 'n': self.n}


@keras.src.saving.register_keras_serializable(package="ArcFace")
class L2Normalization(Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "axis": self.axis}


class MobileFaceNetEmbedding(Model):
    def __init__(self, embedding_size=128, n_classes=85_742, input_shape=(INPUT_W,INPUT_H,INPUT_CHANNELS), **kwargs):
        super(MobileFaceNetEmbedding, self).__init__(**kwargs)

        self.embedding_size = embedding_size
        self.n_classes = n_classes

        self.conv3x3 =  Conv2D(64, 3, strides=2, padding='same', use_bias=False, kernel_regularizer=l2(4e-5), kernel_initializer="he_normal")
        self.bn1 = BatchNormalization()
        self.prelu1 = PReLU(shared_axes=[1,2])

        self.dwise3x3 = DepthwiseConv2D(3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(4e-5), depthwise_initializer="he_normal")
        self.bn2 = BatchNormalization()
        self.prelu2 = PReLU(shared_axes=[1,2])

        self.bottleneck_block_list = []
        self.bottleneck_block_list.append(BottleneckBlock(filters=64, t=2, strides=2, n=5))
        self.bottleneck_block_list.append(BottleneckBlock(filters=128, t=4, strides=2, n=1))
        self.bottleneck_block_list.append(BottleneckBlock(filters=128, t=2, strides=1, n=6))
        self.bottleneck_block_list.append(BottleneckBlock(filters=128, t=4, strides=2, n=1))
        self.bottleneck_block_list.append(BottleneckBlock(filters=128, t=2, strides=1, n=2))

        self.conv1x1 = Conv2D(512, 1, use_bias=False, kernel_regularizer=l2(5e-4), kernel_initializer="he_normal")
        self.bn3 = BatchNormalization()
        self.prelu3 = PReLU(shared_axes=[1,2])

        self.GDconv7x7 = DepthwiseConv2D(7,strides=1  ,padding='valid', use_bias=False, depthwise_regularizer=l2(4e-4), depthwise_initializer="he_normal")
        self.bn4 = BatchNormalization()
        self.prelu4 = PReLU(shared_axes=[1,2])

        self.conv1x1_2 = Conv2D(embedding_size, 1,strides=1 ,use_bias=False, kernel_regularizer=l2(4e-4), kernel_initializer="he_normal")
        self.flatten = Flatten()
        self.embedding = L2Normalization(axis=1)
        self.arcface_layer = ArcFace(n_classes=self.n_classes, s=30.0, name="arcface")


        input = Input(shape=input_shape)
        self.call(input)

        labels = Input(shape=(self.n_classes,))
        embeddings = Input(shape=(self.embedding_size,))

        self.arcface_layer.build(input_shape=[embeddings.shape, labels.shape])

    def build(self, input_shape):
        super(MobileFaceNetEmbedding, self).build(input_shape)


    def call(self, inputs, training=False):
        x = self.conv3x3(inputs)
        x = self.bn1(x,training=training)
        x = self.prelu1(x)

        x = self.dwise3x3(x)
        x = self.bn2(x,training=training)
        x = self.prelu2(x)

        for block in self.bottleneck_block_list:
            x = block(x, training=training)

        x = self.conv1x1(x)
        x = self.bn3(x,training=training)
        x = self.prelu3(x)

        x = self.GDconv7x7(x)
        x = self.bn4(x,training=training)
        x = self.prelu4(x)

        x = self.conv1x1_2(x)
        x = self.flatten(x)
        x = self.embedding(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'embedding_size': self.embedding_size, 'n_classes': self.n_classes}

    def compute_output_shape(self, input_shape):
        return None, self.embedding_size

    def train_step(self, data):
        X, y = data
        y  = tf.one_hot(y, depth=self.n_classes)
        with tf.GradientTape() as tape:
            embeddings = self(X, training=True)
            logits = self.arcface_layer([embeddings, y])
            loss = self.loss(y, logits)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        y_pred = tf.nn.softmax(logits)
        for metric in self.metrics:
            metric.update_state(y, y_pred)
        return {"arcface_loss": loss, **{metric.name: metric.result() for metric in self.metrics}}

    def test_step(self, data):
        X, y = data
        y  = tf.one_hot(y, depth=self.n_classes)
        embeddings = self(X, training=False)
        y_pred = self.arcface_layer([embeddings, y])
        loss = self.loss(y, y_pred)
        print(self.metrics)
        for metric in self.metrics:
            metric.update_state(y, y_pred)

        return {"arcface_loss": loss, **{metric.name: metric.result() for metric in self.metrics}}


class MarginCallback(Callback):
    def __init__(self, scheduler):
        super(MarginCallback, self).__init__()
        self.scheduler = scheduler

    def on_epoch_begin(self, epoch, logs=None):
        margin = self.scheduler(epoch)
        if hasattr(self.model.arcface_layer, "m"):
            self.model.arcface_layer.m.assign(margin)
            print("margin is set to ", margin)


def lr_scheduler(epoch, lr):
    if  epoch == 25:
        return lr / 10 # 0.01
    elif epoch == 50:
        return lr / 10 # 0.001
    elif epoch == 65:
        return lr / 10 # 0.0001
    else:
        return lr

def margin_scheduler(epoch):
    if epoch < 15: # softmax pre-training
        return 0
    return 0.5




def main():
     # create_shuffled_tfrecord("../ms1m-arcface",
     #                         "../ms1m-arcface/debug_dataset.tfrecord",
     #                         debug=True,
     #                         shuffle=True)
    dataset = load_dataset(dataset_path,batch_size=128,shuffle_buffer=256)
    print(dataset)
    model = MobileFaceNetEmbedding(embedding_size=128,n_classes=85_742)
    print(model.summary())

    lr_decay_clb = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    checkpoint_clb = tf.keras.callbacks.ModelCheckpoint("../Checkpoints/{epoch:02d}-{accuracy:.2f}-{loss:.5}.keras",
                                                        monitor='accuracy',
                                                        verbose=1,
                                                        save_best_only=False,
                                                        save_weights_only=False,
                                                        save_freq="epoch",
                                                        mode='max')
    margin_clb = MarginCallback(margin_scheduler)
    model.compile(optimizer= tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"), tf.keras.metrics.Precision(name="precision")],
                  loss = CategoricalCrossentropy(from_logits=True))
    model.build(input_shape=(None, INPUT_W, INPUT_H, INPUT_CHANNELS))
    with tf.device("/GPU:0"):
        model.fit(dataset, epochs=70, callbacks=[lr_decay_clb, checkpoint_clb, margin_clb], initial_epoch=1)
    model.save("../Model_Weights/ArcFace-MobileFaceNet.keras")
    print("Model Saved")



if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()