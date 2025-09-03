import numpy as np
import platform
import cv2
import pickle

if platform.system() == "Linux":
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter


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


class LiteIdentifier(object):
    def __init__(self, lite_detection_path, lite_recognition_path, embeddings_path):
        self.detection_interpreter = Interpreter(model_path=lite_detection_path)
        self.detection_interpreter.allocate_tensors()
        self.detection_input_details = self.detection_interpreter.get_input_details()
        self.detection_output_details = self.detection_interpreter.get_output_details()

        self.recognition_interpreter = Interpreter(model_path=lite_recognition_path)
        self.recognition_interpreter.allocate_tensors()
        self.recognition_input_details = self.recognition_interpreter.get_input_details()
        self.recognition_output_details = self.recognition_interpreter.get_output_details()

        with open(embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)

        ds_embeddings = np.array(list(self.embeddings.values()), dtype=np.float32)
        self.ds_embeddings = np.reshape(ds_embeddings, (-1, ds_embeddings.shape[-1]))

    def identify(self, frame):
        H, W = frame.shape[:2]

        pre = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pre = cv2.resize(pre, (640, 640))
        pre = pre.astype(np.float32)
        pre = pre - 127.5
        pre = pre / 128.0
        pre = np.expand_dims(pre, axis=0)

        self.detection_interpreter.set_tensor( self.detection_input_details[0]['index'], pre)
        self.detection_interpreter.invoke()

        boxes = self.detection_interpreter.get_tensor(self.detection_output_details[1]['index'])
        landmarks = self.detection_interpreter.get_tensor(self.detection_output_details[0]['index'])
        scores = self.detection_interpreter.get_tensor(self.detection_output_details[2]['index'])
        mask = scores >= 0.5
        boxes = boxes[mask]
        landmarks = landmarks[mask]

        identities = []
        for box, landmark in zip(boxes, landmarks):
            x1, x2, x3, x4, x5 = landmark[0::2] * W
            y1, y2, y3, y4, y5 = landmark[1::2] * H
            box *= [H,W,H,W]
            landmarks = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]])

            aligned = align_face(frame, landmarks)
            pre = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            pre = pre.astype(np.float32)
            pre = pre - 127.5
            pre = pre / 128.0
            pre = np.expand_dims(pre, axis=0)

            self.recognition_interpreter.set_tensor(self.recognition_input_details[0]['index'], pre)
            self.recognition_interpreter.invoke()

            embedding = self.recognition_interpreter.get_tensor(self.recognition_output_details[0]["index"])[0]

            cosine_similarity = np.matmul(self.ds_embeddings, embedding)

            idx_max = np.argmax(cosine_similarity)
            person_idx = int(idx_max // 30)
            person = list(self.embeddings.keys())[person_idx]
            if cosine_similarity[idx_max] >= 0.5:
                identities.append([person, box, landmarks])
            else:
                identities.append(["unknown", box, landmarks])
        return identities

