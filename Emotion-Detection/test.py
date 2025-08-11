'''
PyPower Projects
Emotion Detection Using AI
'''

# USAGE: python test.py

import cv2
import numpy as np
from time import sleep

# ✅ Use tf.keras instead of standalone keras
from tensorflow.keras.models import load_model

# Load face detector and emotion classification model
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')

# Emotion labels (make sure they match the training order)
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open the webcam (use 1 instead of 0 if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        # Could not read a frame from the camera
        continue

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(frame, 'No Face Found', (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (face) and preprocess
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float32') / 255.0  # Normalize to [0,1]
        roi = np.expand_dims(roi, axis=-1)        # Shape: (48, 48, 1)
        roi = np.expand_dims(roi, axis=0)         # Shape: (1, 48, 48, 1)

        # Predict emotion
        preds = classifier.predict(roi)[0]        # Shape: (num_classes,)
        label_index = int(np.argmax(preds))
        label = class_labels[label_index]

        # Debugging logs
        print(f"\nprediction = {preds}")
        print(f"prediction max = {label_index}")
        print(f"label = {label}\n")

        # Write the predicted label on the frame
        label_position = (x, y - 10 if y - 10 > 20 else y + h + 30)
        cv2.putText(frame, label, label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show the video feed with predictions
    cv2.imshow('Emotion Detector', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
