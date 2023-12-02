# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:29:11 2023

@author: okzki
"""

import cv2

cam_acc = cv2.VideoCapture(0) #captura la camara

while True:
    ind, frame = cam_acc.read() #lee frame de la imagen uno por uno
    cv2.imshow('frame', frame) #arroja frame/img
    
    key = cv2.waitKey(1) #espera hasta que se presiona la tecla
    if key == ord("s"): #se para la captura al presionar s
        break
    
cam_acc.release() # saca el vid
cv2.destroyAllWindows() #quita todas las "frame"


    

