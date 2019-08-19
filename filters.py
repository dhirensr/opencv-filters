# Turban filters using OpenCV
# @author:- Dhiren Serai (dhirensr)

import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
import os
import subprocess

cascPath = "haarcascade_frontalface_default.xml"  # for face detection

if not os.path.exists(cascPath):
    subprocess.call(['./download_filters.sh'])
else:
    print('Filters already exist!')

faceCascade = cv2.CascadeClassifier(cascPath)


video_capture = cv2.VideoCapture(0)

turban=cv2.imread('turban.jpg')

def put_turban(turban,fc,x,y,w,h):

    face_width = w
    face_height = h

    hat_width = face_width+1
    hat_height = int(0.65*face_height)+1

    hat = cv2.resize(turban,(hat_width,hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k]<235:
                    fc[y+i-int(0.25*face_height)][x+j][k] = hat[i][j][k]
    return fc



while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40,40)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText(frame,"Person Detected",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        frame = put_turban(turban,frame,x,y,w,h)

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
