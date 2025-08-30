import tensorflow as tf
from AnchorBoxes import *
from dataset import load_dataset
from MobilenetV3 import *
from losses import *
from tensorflow.keras.applications import MobileNetV3Large




def lr_scheduler(epoch, lr):
    if epoch < 5:
        return 0.001

    elif epoch < 55:
        return 0.01

    elif epoch < 68:
        return 0.001
    else:
        return 0.0001



def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    dataset = load_dataset("../WIDER_train/train.tfrecord", batch_size=7, shuffle_buffer=3)

    anchor_generator = AnchorBoxes(box_sizes=[[16, 32], [64, 128], [256, 512]], image_size=(640, 640),
                                   steps=[8, 16, 32])
    boxes = anchor_generator.generate_anchors()
    boxes = anchor_generator.normalize_anchors(boxes)

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

    detector = Detector(backbone)
    detector.backbone.trainable = True
    dummy = tf.random.normal((1, 640, 640, 3))
    _ = detector(dummy)
    print(detector.summary(expand_nested=False))


    checkpoint_clb = tf.keras.callbacks.ModelCheckpoint("./Checkpoints/{epoch:02d}.keras",
                                                        verbose=1,
                                                        save_best_only=False,
                                                        save_weights_only=False,
                                                        save_freq="epoch",
                                                        mode='max')


    lr_clb = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    detector.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, clipnorm=1.0) ,
                     cls_loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction="none"),
                     reg_loss=Smooth_L1(delta=1.0, name="reg_loss"),
                     lnd_loss=Smooth_L1(delta=1.0, name="lnd_loss"),
                     anchors=boxes)

    detector.fit(dataset, epochs=80, callbacks=[checkpoint_clb, lr_clb])
    detector.save("FaceDetector-serialized.keras")









if __name__ == "__main__":
    tf.random.set_seed(42)
    # tf.debugging.set_log_device_placement(True)
    # tf.debugging.enable_check_numerics()
    # tf.config.run_functions_eagerly(True)
    main()