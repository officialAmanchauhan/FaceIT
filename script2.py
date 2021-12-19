import cv2  
  
# Load the cascade  
face_cascade = cv2.CascadeClassifier('frontfacepythonfile.xml')  
  
# To capture video from existing video , here happykid.mp4 is the resource file   
vid = cv2.VideoCapture('happykid.mp4')         
""" 
Here we used a video source since it is also a collection of many frames combined
You can also use your webcam to detect your face just replace ('happykid.mp4') to (0)
"""

while True:                   #Running a loop to read all the frames with flags & frames as _,img
    # Reads the frame  
    _, img = vid.read()  

    # Detect the faces  
    faces = face_cascade.detectMultiScale(img, 1.1, 4)  
  
    # Draw the rectangle around each face  
    for (x, y, w, h) in faces:  
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  
  
    # Display  
    cv2.imshow('Video', img)  
  
    # Since it is a never ending loop , you can stop it if escape key is pressed  from this code written below
    k = cv2.waitKey(1) 
    if k==27:  
        break  
    
    """ 
    Here, waitkey is a function that allows user to display a window for a given set of time
    """
    
# Release the VideoCapture object  
vid.release()  

