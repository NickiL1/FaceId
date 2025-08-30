
import keras.src.saving
import tensorflow as tf
from keras import Layer
from tensorflow.keras.layers import (Conv2D, BatchNormalization, PReLU,
                                      UpSampling2D,
                                     Softmax)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from Detection.losses import *
from tensorflow.keras.applications import MobileNetV3Large
from Detection.AnchorBoxes import AnchorBoxes, Iou
import time


@keras.src.saving.register_keras_serializable(package='RetinaFace')
class FPN(Model):
    def __init__(self, out_channels, **kwargs):
        super(FPN, self).__init__(**kwargs)
        self.out_channels = out_channels

        self.upsample = UpSampling2D(size=(2, 2), interpolation="bilinear")

        self.conv1x1_p5 = Conv2D(self.out_channels, 1, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.bn_p5 = BatchNormalization()
        self.prelu_p5 = PReLU(shared_axes=[1, 2],alpha_initializer=tf.keras.initializers.Constant(0.1)
)

        self.conv1x1_p4 = Conv2D(self.out_channels, 1, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.bn_1x1_p4 = BatchNormalization()
        self.prelu_1x1_p4 = PReLU(shared_axes=[1, 2],alpha_initializer=tf.keras.initializers.Constant(0.1)
)

        self.conv3x3_p4 = Conv2D(self.out_channels, 3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.bn_3x3_p4 = BatchNormalization()
        self.prelu_3x3_p4 = PReLU(shared_axes=[1, 2],alpha_initializer=tf.keras.initializers.Constant(0.1)
)

        self.conv1x1_p3 = Conv2D(self.out_channels, 1, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.bn_1x1_p3 = BatchNormalization()
        self.prelu_1x1_p3 = PReLU(shared_axes=[1, 2],alpha_initializer=tf.keras.initializers.Constant(0.1)
)

        self.conv3x3_p3 = Conv2D(self.out_channels, 3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.bn_3x3_p3 = BatchNormalization()
        self.prelu_3x3_p3 = PReLU(shared_axes=[1, 2],alpha_initializer=tf.keras.initializers.Constant(0.1)
)

    def build(self, input_shape):
        super(FPN, self).build(input_shape)

    def call(self, inputs, training=False):
        c3, c4, c5 = inputs

        p5 = self.conv1x1_p5(c5)
        p5 = self.bn_p5(p5, training=training)
        p5 = self.prelu_p5(p5)

        p4_lat = self.conv1x1_p4(c4)
        p4_lat = self.bn_1x1_p4(p4_lat, training=training)
        p4_lat = self.prelu_1x1_p4(p4_lat)
        p4_up = self.upsample(p5)
        p4 = tf.add(p4_up, p4_lat)
        p4 = self.conv3x3_p4(p4)
        p4 = self.bn_3x3_p4(p4, training=training)
        p4 = self.prelu_3x3_p4(p4)

        p3_lat = self.conv1x1_p3(c3)
        p3_lat = self.bn_1x1_p3(p3_lat, training=training)
        p3_lat = self.prelu_1x1_p3(p3_lat)
        p3_up = self.upsample(p4)
        p3 = tf.add(p3_up, p3_lat)
        p3 = self.conv3x3_p3(p3)
        p3 = self.bn_3x3_p3(p3, training=training)
        p3 = self.prelu_3x3_p3(p3)

        return p3, p4, p5

    def get_config(self):
        base = super().get_config()
        return {**base, "out_channels": self.out_channels}


@keras.src.saving.register_keras_serializable(package='RetinaFace')
class SSH(Model):
    def __init__(self, **kwargs):
        super(SSH, self).__init__(**kwargs)


    def build(self, input_shape):
        channels = input_shape[-1]
        self.conv1 = Conv2D(channels // 2, 3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.bn_1 = BatchNormalization()


        self.conv2_1 = Conv2D(channels // 4, 3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.bn_2_1 = BatchNormalization()
        self.prelu2 = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.Constant(0.1))

        self.conv2_2 = Conv2D(channels // 4, 3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.bn_2_2 = BatchNormalization()


        self.conv3_1 = Conv2D(channels // 4, 3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.bn_3_1 = BatchNormalization()
        self.prelu3_1 = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.Constant(0.1))

        self.conv3_2 = Conv2D(channels // 4, 3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.bn_3_2 = BatchNormalization()
        self.prelu3_2 = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.Constant(0.1))

        self.conv3_3 = Conv2D(channels // 4, 3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.bn_3_3 = BatchNormalization()


        super(SSH, self).build(input_shape)

    def call(self, inputs, training=False):
        b1 = self.conv1(inputs)
        b1 = self.bn_1(b1, training=training)


        b2 = self.conv2_1(inputs)
        b2 = self.bn_2_1(b2, training=training)
        b2 = self.prelu2(b2)

        b2 = self.conv2_2(b2)
        b2 = self.bn_2_2(b2, training=training)

        b3 = self.conv3_1(inputs)
        b3 = self.bn_3_1(b3, training=training)
        b3 = self.prelu3_1(b3)

        b3 = self.conv3_2(b3)
        b3 = self.bn_3_2(b3, training=training)
        b3 = self.prelu3_2(b3)

        b3 = self.conv3_3(b3)
        b3 = self.bn_3_3(b3, training=training)

        final = tf.concat([b1, b2, b3], axis=-1)
        return final




@keras.saving.register_keras_serializable(package="RetinaFace")
class Detector(Model):
    def __init__(self, backbone, **kwargs):
        super(Detector, self).__init__(**kwargs)
        self.backbone = backbone
        self.softmax = Softmax(axis=-1)
        self.fpn = FPN(96)
        self.ssh1 = SSH()
        self.ssh2 = SSH()
        self.ssh3 = SSH()




        self.cls_head1 = Conv2D(4, 1, padding='same', kernel_initializer="glorot_uniform", kernel_regularizer=l2(4e-5))
        self.cls_head2 = Conv2D(4, 1, padding='same', kernel_initializer="glorot_uniform", kernel_regularizer=l2(4e-5))
        self.cls_head3 = Conv2D(4, 1, padding='same', kernel_initializer="glorot_uniform", kernel_regularizer=l2(4e-5))

        self.reg_head1 = Conv2D(8, 1, padding='same', kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.reg_head2 = Conv2D(8, 1, padding='same', kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.reg_head3 = Conv2D(8, 1, padding='same', kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))

        self.landmark1 = Conv2D(20, 1, padding='same', kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.landmark2 = Conv2D(20, 1, padding='same', kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))
        self.landmark3 = Conv2D(20, 1, padding='same', kernel_initializer="he_normal", kernel_regularizer=l2(4e-5))

    def build(self, input_shape):
        super(Detector, self).build(input_shape)


    def call(self, inputs, training=False):

        c3, c4, c5 = self.backbone(inputs, training=training)
        p3, p4, p5 = self.fpn([c3, c4, c5], training=training)




        p3 = self.ssh1(p3, training=training)
        p4 = self.ssh2(p4, training=training)
        p5 = self.ssh3(p5, training=training)

        A1, A2, A3 = 80 * 80 * 2, 40 * 40 * 2, 20 * 20 * 2

        cls1 = self.cls_head1(p3)
        cls1 = tf.reshape(cls1, (-1, A1, 2))
        cls1 = self.softmax(cls1)

        cls2 = self.cls_head2(p4)
        cls2 = tf.reshape(cls2, (-1, A2, 2))
        cls2 = self.softmax(cls2)

        cls3 = self.cls_head3(p5)
        cls3 = tf.reshape(cls3, (-1, A3, 2))
        cls3 = self.softmax(cls3)

        reg1 = self.reg_head1(p3)
        reg1 = tf.reshape(reg1, (-1, A1, 4))

        reg2 = self.reg_head2(p4)
        reg2 = tf.reshape(reg2, (-1, A2, 4))

        reg3 = self.reg_head3(p5)
        reg3 = tf.reshape(reg3, (-1, A3, 4))

        landmark1 = self.landmark1(p3)
        landmark1 = tf.reshape(landmark1, (-1, A1, 10))

        landmark2 = self.landmark2(p4)
        landmark2 = tf.reshape(landmark2, (-1, A2, 10))

        landmark3 = self.landmark3(p5)
        landmark3 = tf.reshape(landmark3, (-1, A3, 10))

        cls = tf.concat([cls1,cls2,cls3], axis=1)
        reg  = tf.concat([reg1,reg2,reg3], axis=1)
        landmark = tf.concat([landmark1,landmark2,landmark3], axis=1)

        return cls, reg, landmark

    def compile(self, cls_loss, reg_loss, lnd_loss, anchors, **kwargs):
        super(Detector, self).compile(**kwargs)
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.lnd_loss = lnd_loss
        self.anchors = anchors



    def train_step(self, data):
        image, cls_true, reg_targets_true, lnd_targets_true, lnd_mask_true = data


        reg_mask = tf.cast(tf.equal(cls_true, 1), tf.float32)
        reg_mask = tf.reshape(reg_mask, (-1,))
        lnd_mask = tf.cast(lnd_mask_true, tf.float32)
        lnd_mask = tf.reshape(lnd_mask, (-1,))



        with tf.GradientTape() as tape:
            cls_pred, reg_pred, lnd_pred = self(image, training=True)
            cls_pred = tf.reshape(cls_pred, (-1, 2))
            reg_pred = tf.reshape(reg_pred, (-1, 4))
            lnd_pred = tf.reshape(lnd_pred, (-1, 10))


            #Hard negative mining
            cls_true = tf.reshape(cls_true, (-1,))
            pos_mask = tf.cast(tf.equal(cls_true, 1), tf.float32)
            neg_mask = tf.cast(tf.equal(cls_true, 0), tf.float32)
            num_pos = tf.reduce_sum(pos_mask)
            cls_true = tf.one_hot(cls_true, depth=2)
            cls_loss = self.cls_loss(cls_true, cls_pred)
            cls_loss_pos = tf.multiply(cls_loss, pos_mask)
            cls_loss_pos = tf.reduce_sum(cls_loss_pos)
            cls_loss_neg = tf.multiply(cls_loss, neg_mask)


            k = tf.cast(3 * num_pos, dtype=tf.int32)
            idx = tf.argsort(cls_loss_neg, axis=-1, direction='DESCENDING')
            valid_idx_mask = tf.cast(tf.range(tf.shape(idx)[0]) < k, tf.float32)
            hard_neg_mask = tf.scatter_nd(
                tf.expand_dims(idx, 1),
                valid_idx_mask,
                [tf.shape(idx)[0]]
            )


            cls_loss_neg = tf.multiply(cls_loss_neg, hard_neg_mask)
            cls_loss_neg = tf.reduce_sum(cls_loss_neg)
            cls_loss = (cls_loss_pos + cls_loss_neg) / (num_pos)



            reg_targets_true = tf.reshape(reg_targets_true, (-1, 4))
            reg_loss = self.reg_loss.call(reg_targets_true, reg_pred)
            reg_mask = tf.expand_dims(reg_mask, axis=-1)
            reg_loss = tf.multiply(reg_loss, reg_mask)
            reg_loss = tf.reduce_sum(reg_loss) / (tf.reduce_sum(reg_mask) + 1e-6)



            lnd_targets_true = tf.reshape(lnd_targets_true, (-1, 10))
            lnd_loss = self.lnd_loss.call(lnd_targets_true, lnd_pred)
            lnd_mask = tf.expand_dims(lnd_mask, axis=-1)
            lnd_loss = tf.multiply(lnd_loss, lnd_mask)
            lnd_loss = tf.reduce_sum(lnd_loss) / (tf.reduce_sum(lnd_mask) + 1e-6)


            total_loss = reg_loss + lnd_loss + cls_loss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


        return {"loss": total_loss, "reg_loss": reg_loss, "lnd_loss": lnd_loss, "cls_loss": cls_loss}

    def get_config(self):
        config = super().get_config()
        config.update({
            "backbone": self.backbone
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["backbone"] = keras.saving.deserialize_keras_object(config["backbone"])
        return cls(**config)


@keras.saving.register_keras_serializable(package="RetinaFace")
class NMS(Layer):
    def __init__(self, nms_thresh, anchors, variances=[0.1,0.2], **kwargs):
        super(NMS, self).__init__(**kwargs)
        self.nms_thresh = nms_thresh
        self.anchors = anchors
        self.variances = variances

    def call(self, inputs, training=False):
        cls_pred, reg_pred, lnd_pred = inputs

        # Classification
        cls_pred = tf.reshape(cls_pred, (-1, 2))[:, 1]

        # Regression (bbox)
        reg_pred = tf.reshape(reg_pred, (-1, 4))
        reg_pred = reg_pred * tf.constant([self.variances[0], self.variances[0], self.variances[1], self.variances[1]], dtype=tf.float32)

        dx, dy, dw, dh = tf.unstack(reg_pred, axis=1)
        x_a, y_a, w_a, h_a = tf.unstack(self.anchors, axis=1)

        xc_pred = dx * w_a + x_a
        yc_pred = dy * h_a + y_a
        w_pred = tf.exp(dw) * w_a
        h_pred = tf.exp(dh) * h_a
        xmin = xc_pred - w_pred / 2
        ymin = yc_pred - h_pred / 2
        xmax = xc_pred + w_pred / 2
        ymax = yc_pred + h_pred / 2
        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)


        # Landmarks
        lnd_pred = tf.reshape(lnd_pred, (-1, 5, 2))
        lnd_pred = lnd_pred * self.variances[0]

        # scale and shift: (dx * w_a + x_a, dy * h_a + y_a)
        x_coords = lnd_pred[..., 0] * w_a[:, None] + x_a[:, None]
        y_coords = lnd_pred[..., 1] * h_a[:, None] + y_a[:, None]

        lnd_pred = tf.reshape(tf.stack([x_coords, y_coords], axis=-1), (-1, 10))

        selected_idx = tf.image.non_max_suppression(
            boxes,
            cls_pred,
            max_output_size=200,
            iou_threshold=self.nms_thresh,
            score_threshold=self.nms_thresh,
        )
        return tf.gather(boxes,selected_idx), tf.gather(lnd_pred,selected_idx)

    def get_config(self):
        base = super(NMS, self).get_config()
        return {**base, "nms_thresh": self.nms_thresh, "anchors": self.anchors, "variances": self.variances}






@keras.saving.register_keras_serializable(package="RetinaFace")
class FaceDetector(Model):
    def __init__(self,weights_path, **kwargs):
        super(FaceDetector, self).__init__(**kwargs)
        self.weights_path = weights_path

        backbone = MobileNetV3Large(
            input_shape=(640, 640, 3),
            include_top=False,
            weights="imagenet",
            pooling=None,
            name="MobileNetV3Large",
            include_preprocessing=False
        )
        c3 = backbone.get_layer("expanded_conv_5_add").output
        c4 = backbone.get_layer("expanded_conv_11_add").output
        c5 = backbone.get_layer("expanded_conv_14_add").output

        backbone = Model(inputs=backbone.input, outputs=[c3, c4, c5], name="Backbone")
        self.raw_detector = Detector(backbone)
        dummy = tf.random.normal((1, 640, 640, 3))
        _ = self.raw_detector(dummy)
        self.raw_detector.load_weights(weights_path)

        anchor_generator = AnchorBoxes(box_sizes=[[16, 32], [64, 128], [256, 512]], image_size=(640, 640),
                                       steps=[8, 16, 32])
        boxes = anchor_generator.generate_anchors()
        self.boxes = anchor_generator.normalize_anchors(boxes)

        self.nms = NMS(0.5, self.boxes)


    def call(self, inputs, training=False):
        cls, reg, lnd = self.raw_detector(inputs, training=False)
        boxes, landmarks = self.nms([cls, reg, lnd])
        return boxes, landmarks

    def get_config(self):
        base = super(FaceDetector, self).get_config()
        return {**base, "weights_path": self.weights_path}





