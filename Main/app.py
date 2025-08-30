from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import threading
import time
from Detection.MobilenetV3 import FaceDetector
from Recognition.train_arcface import MobileFaceNetEmbedding
from Identifier import *

app = FastAPI()


frame = None
running = True



camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

identifier = FaceIdentifier("../Model_Weights/FaceDetector-serialized.keras",
                           "../Model_Weights/ArcFace-MobileFaceNet.keras",
                           "../Database")



def capture_loop():

    global frame, running
    while running:
        success, img = camera.read()
        if img is not None and success is True:
            result = identifier.identify(img)

            for identity in result:
                person = identity[0]
                bbox = identity[1]
                landmarks = identity[2]

                ymin, xmin, ymax, xmax = bbox

                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

                cv2.putText(
                    img,
                    person,
                    (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                for (x, y) in landmarks:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), 3)
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()


threading.Thread(target=capture_loop, daemon=True).start()

def generate_frames():
    global frame, running
    while running:
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.01)


@app.get("/")
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.on_event("shutdown")
def shutdown_event():
    global running
    running = False
    camera.release()
    print("Camera released, shutdown complete.")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)