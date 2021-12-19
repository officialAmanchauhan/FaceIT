#!/usr/bin/python3
import cv2

"""
OpenCV is an Open Source Computer Vision Library used for machine learning for performing real-time operations
With OpenCV, you can do many things
-> Face Detection
-> Face Recognition
-> Object Recognition
-> Motion Tracking
-> Mobile Robotics
-> Statistical machine learning 
and many more functionalities and can be used for security, emergency , verification , advance ticketing systems and many many more purposes
OpenCV has many pre-trained classifiers for face, eyes, smile , etc on https://github.com/opencv/opencv/tree/master/data/haarcascades
"""

def faceDetect():                                                       # Defining function faceDetect

    face_finder = cv2.CascadeClassifier('frontfacepythonfile.xml')
    
    """
    Cascade Clasifier is used to detect object in images
    I have used front face classifier  and specified it as frontfacepythonfile.xml
    """
    
    img = cv2.imread('monalisa.jpg')                # Reads the image from the file specified by filename, Here monalisa.jpg
 
    faces = face_finder.detectMultiScale(img, 1.1, 4)  
    
    """
    This detect faces with three arguments (image , scale factor, Minneighbors )
    Scale factor specifies how much image size is reduced with each scale
    MinNeighbors specifies how many neighbors each candidate rectangle should have to retain it general is around 4-5
    Note: Here, I didn't convert image to grayscale. I found OpenCV can detect faces from still images and can be done same with videos
    if theya re converted to frames first that i have done in script2
    """

    for (x, y, w, h) in faces: 
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 10)
        
        """ 
        Now we are drawing a rectangle to the detected faces with coordinates x,y,w & h with RBG value (255,0,0)(Blue) with thickness !0
        """
    
    return cv2.imwrite("monalisaface_detected.png", img)             # Returning value as image ( it's optional we can get image without calling our function back , it is just I Like it)

faceDetect()                                 # Calling faceDetect Function to detect faces in given samples