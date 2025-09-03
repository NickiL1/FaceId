import cv2
import numpy as np
import time
import pickle
import os
import shutil
import tensorflow as tf


def align_face(img, landmarks, output_size=(112, 112)):

    # Standard 5-point landmark template (ArcFace/InsightFace style)
    template = np.array([
        [30.2946, 51.6963],   # left eye
        [65.5318, 51.5014],   # right eye
        [48.0252, 71.7366],   # nose tip
        [33.5493, 92.3655],   # left mouth
        [62.7299, 92.2041]    # right mouth
    ], dtype=np.float32)

    if output_size[0] == 112:  # small adjustment for 112x112
        template[:, 0] += 8.0

    # Convert inputs
    src = np.array(landmarks, dtype=np.float32)
    dst = template

    # Estimate similarity transform
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    # Apply transform
    aligned_face = cv2.warpAffine(img, M, output_size, borderValue=0)

    return aligned_face


class DB_Manager(object):
    def __init__(self, dir_path, detection_model, embedding_model):
        super(DB_Manager, self).__init__()
        self.dir_path = dir_path
        self.emb_path = os.path.join(self.dir_path, "Embeddings.pkl")
        self.embedding_model = embedding_model
        self.detection_model = detection_model

        if os.path.exists(self.emb_path):
            with open(self.emb_path, "rb") as f:
                self.embeddings = pickle.load(f)
        else:
            self.embeddings = {}



    def add_person(self, person, override=False):
        if override is False and person in self.embeddings:
            print("person already added, and override is False")
            return
        embeddings = []
        person_dir = os.path.join(self.dir_path, person)
        if  not os.path.isdir(os.path.join(person_dir)):
            os.makedirs(person_dir)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                print(f" Failed to grab frame {i}")
                break
            H, W = frame.shape[:2]
            pre = tf.reverse(frame, [-1])
            pre = tf.image.resize(pre, (640, 640))
            pre = tf.cast(pre, tf.float32)
            pre = tf.subtract(pre, 127.5,)
            pre = tf.multiply(pre, 0.0078125)
            pre = tf.expand_dims(pre, axis=0)

            bbox, lnds = self.detection_model.predict(pre, batch_size=1)
            x1, x2, x3, x4, x5 = lnds[0][0::2] * W
            y1, y2, y3, y4, y5 = lnds[0][1::2] * H
            landmarks = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]])
            aligned = align_face(frame, landmarks)

            cv2.imshow(person, aligned)
            filename = os.path.join(person_dir, f"frame_{i:02d}.jpg")
            cv2.imwrite(filename, aligned)


            pre = tf.reverse(aligned, [-1])
            pre = tf.cast(pre, tf.float32)
            pre = tf.subtract(pre, 127.5,)
            pre = tf.multiply(pre, 0.0078125)
            pre = tf.expand_dims(pre, axis=0)
            embedding = self.embedding_model.predict(pre, batch_size=1)[0]
            embeddings.append(embedding)
            # Delay
            cv2.waitKey(500)

        self.embeddings[person] = embeddings
        with open(self.emb_path, "wb") as f:
            pickle.dump(self.embeddings, f)

    def remove_person(self, person):
        if person in self.embeddings:
            del self.embeddings[person]
            with open(self.emb_path, "wb") as f:
                pickle.dump(self.embeddings, f)
        if os.path.isdir(os.path.join(self.dir_path, person)):
            shutil.rmtree(os.path.join(self.dir_path, person))

    def get_embeddings(self):
        return self.embeddings





class FaceIdentifier(object):
    def __init__(self, detection_weights, recognition_weights, database_path):
        self.detection_weights = detection_weights
        self.recognition_weights = recognition_weights
        self.database_path = database_path

        self.detection_model = tf.keras.models.load_model(self.detection_weights)
        self.recognition_model = tf.keras.models.load_model(self.recognition_weights)
        self.database = DB_Manager(self.database_path, self.detection_model, self.recognition_model)

        self.embeddings = self.database.get_embeddings()

    def identify(self, frame):
        H, W = frame.shape[:2]
        if len(self.embeddings) == 0:
            print("No embeddings found")
            return []

        pre = tf.reverse(frame, [-1])
        pre = tf.image.resize(pre, (640, 640))
        pre = tf.cast(pre, tf.float32)
        pre = tf.subtract(pre, 127.5,)
        pre = tf.multiply(pre, 0.0078125)
        pre = tf.expand_dims(pre, axis=0)

        bbox, lnds, scores = self.detection_model.predict(pre, batch_size=1)

        identities = []
        for box, landmark in zip(bbox, lnds):
            x1, x2, x3, x4, x5 = landmark[0::2] * W
            y1, y2, y3, y4, y5 = landmark[1::2] * H
            box *= [H,W,H,W]
            landmarks = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]])
            aligned = align_face(frame, landmarks)

            pre = tf.reverse(aligned, [-1])
            pre = tf.cast(pre, tf.float32)
            pre = tf.subtract(pre, 127.5, )
            pre = tf.multiply(pre, 0.0078125)
            pre = tf.expand_dims(pre, axis=0)

            embedding = self.recognition_model.predict(pre, batch_size=1)[0]

            ds_embeddings =  tf.convert_to_tensor(list(self.embeddings.values()), dtype=tf.float32)
            ds_embeddings = tf.reshape(ds_embeddings, (-1, ds_embeddings.shape[-1]))
            cosine_similarity = tf.matmul(ds_embeddings, tf.expand_dims(embedding, -1), transpose_b=False)

            idx_max = tf.argmax(cosine_similarity)[0]
            person_idx = int(idx_max // 30)
            person = list(self.embeddings.keys())[person_idx]
            if cosine_similarity[idx_max] >= 0.5:
                identities.append([person, box, landmarks])
            else:
                identities.append(["unknown", box, landmarks])
        return identities

