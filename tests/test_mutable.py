import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Create a dummy landmark list
landmarks = landmark_pb2.NormalizedLandmarkList()
lm = landmarks.landmark.add()
lm.x = 0.5
lm.y = 0.5
lm.z = 0.5

print("Before:", lm.x)
try:
    lm.x = 0.8
    print("After:", lm.x)
    print("Mutable!")
except Exception as e:
    print("Error:", e)
