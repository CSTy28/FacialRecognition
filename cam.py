# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 21:30:52 2019

@author: bigfe
"""

import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            print(id)
            print(labels[id_])
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_color)
        
        color = (255, 0, 0)
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x,y), (width, height), color, stroke)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()