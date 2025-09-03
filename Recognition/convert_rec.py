import tensorflow as tf
from Recognition.train_arcface import MobileFaceNetEmbedding


def main():
    model = tf.keras.models.load_model("../Model_Weights/ArcFace-MobileFaceNet.keras")


    dummy = tf.random.uniform(shape=(1,112,112,3))
    _ = model(dummy)
    model.export("recognition", save_format="tf_saved_model")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir="recognition")
    tflite_model = converter.convert()

    with open("../Model_Weights/recognition.tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    main()