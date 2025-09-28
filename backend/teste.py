import tensorflow as tf, mediapipe as mp, numpy as np
from PIL import Image

print("TF:", tf.__version__)               # deve imprimir 2.13.0
print("MediaPipe:", mp.__version__)        # 0.10.x
arr = np.array(Image.new("RGB",(320,240),(128,128,128)))
with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
    print("FaceDetection OK:", fd.process(arr) is not None)
