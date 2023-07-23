#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

face_cascade = cv2.CascadeClassifier('/home/abud/git/yt_js/haar/haarcascade_frontalface_default.xml')

def detect(gray ,image):
    faces =face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(3, 3))
    # print(faces)
    for (x,y ,w ,h) in faces : # her yüz için tanıma yapıyor döngü 
        cv2.rectangle(image , (x,y), (x+w ,y+h) , (255,0,0) ,2 ) 
    return image

im_path = "img/faces.jpg"

w_size = (1024, 512)
img = cv2.imread(im_path, cv2.IMREAD_COLOR)
# img = cv2.resize(img,w_size)
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
k=15
gray = cv2.GaussianBlur(gray, (k, k), 0)
canvas = detect(gray,img)

cv2.namedWindow("out", cv2.WINDOW_NORMAL)
cv2.imshow('out',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
