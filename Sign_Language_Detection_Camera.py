#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd
import time


# In[3]:


# Load trained model
model = load_model('smnist.h5')
letterpred = [chr(i) for i in range(65, 91) if chr(i) != 'J' and chr(i) != 'Z']


# In[4]:


# Initialize MediaPipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils


# In[5]:


# Start webcam
cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, _ = frame.shape

while True:
    ret, frame = cap.read()
    if not ret:
        break
    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC to exit
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:  # SPACE to predict
        analysisframe = frame.copy()
        framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
        if hand_landmarksanalysis:
            for handLMsanalysis in hand_landmarksanalysis:
                x_max, y_max, x_min, y_min = 0, 0, w, h
                for lmanalysis in handLMsanalysis.landmark:
                    x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                    x_max, x_min = max(x_max, x), min(x_min, x)
                    y_max, y_min = max(y_max, y), min(y_min, y)
                y_min, y_max = max(0, y_min-20), min(h, y_max+20)
                x_min, x_max = max(0, x_min-20), min(w, x_max+20)
                analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
                analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                analysisframe = cv2.resize(analysisframe, (28,28))
                pixeldata = analysisframe.reshape(1,28,28,1) / 255.0
                prediction = model.predict(pixeldata)
                predarray = prediction[0]
                top3 = predarray.argsort()[-3:][::-1]
                for i, idx in enumerate(top3):
                    print(f"Predicted Character {i+1}: {letterpred[idx]} (Confidence: {predarray[idx]*100:.2f}%)")
                time.sleep(2)
    # Draw hand landmarks
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max, y_max, x_min, y_min = 0, 0, w, h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max, x_min = max(x_max, x), min(x_min, x)
                y_max, y_min = max(y_max, y), min(y_min, y)
            y_min, y_max = max(0, y_min-20), min(h, y_max+20)
            x_min, x_max = max(0, x_min-20), min(w, x_max+20)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    cv2.imshow("Sign Language Detection", frame)

cap.release()
cv2.destroyAllWindows()


# In[ ]:




