#!/usr/bin/python3
import cv2
def faceDetect():
    face_cascade = cv2.CascadeClassifier('frontfacepythonfile.xml')
    img = cv2.imread('monalisa.jpg')
 
    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    for (x, y, w, h) in faces: 
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 10)
    
    return cv2.imwrite("monalisaface_detected.png", img) 

faceDetect()