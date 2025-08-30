import tensorflow as tf
from tensorflow.keras.losses import Loss



class Smooth_L1(Loss):
    def __init__(self, delta=1.0, name="Smooth_L1", **kwargs):
        super(Smooth_L1, self).__init__(**kwargs)
        self.delta = delta
        self.name = name

    def call(self, y_true, y_pred):
        loss = tf.abs(y_true - y_pred)
        cond = tf.less(loss, self.delta)
        loss = tf.where(cond, 0.5 * tf.square(loss), self.delta * loss - 0.5 * tf.square(self.delta))
        return loss

    def get_config(self):
        return {"delta": self.delta, "name": self.name}


