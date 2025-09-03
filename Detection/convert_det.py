import tensorflow as tf
from MobilenetV3 import FaceDetector

def main():

    model = FaceDetector("../Model_Weights/FaceDetector.keras")
    dummy_input = tf.random.normal((1, 640, 640, 3))
    _ = model(dummy_input)

    model.export("detection", save_format="tf_saved_model")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir="detection")
    tflite_model = converter.convert()

    with open("../Model_Weights/detection.tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    main()