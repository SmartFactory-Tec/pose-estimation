# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:02:46 2023

@author: okzki
"""

import cv2
import numpy as np
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils #variable que nos da para dibujar las poses
mp_pose = mp.solutions.pose #variable que nos da el modelo de estimación de pose

capcam = cv2.VideoCapture(0) #captura la camara


#acceso a la estimación de pose con metricas de que tan preciso son los datos arrojados y mantiene el estado de detección de la img
with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.4) as pose:
    while capcam.isOpened():
        ind, frame = capcam.read() #lectura de la imagen uno por uno
        
        #cambiando color a RGB para que pase a mediapipe en un mejor orden
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        #detección
        result = pose.process(img)# variable de detección de img
        
        #se regresa a BGR
        img = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR) 
        
        #Obtenemos estimaciones x,y,z de las posiciones de los elementos del cuerpo
        try:
            xyz_lndmrk = result.pose_landmarks.landmark
            
            lft_shldr =  [xyz_lndmrk[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,xyz_lndmrk[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            rgt_shldr =  [xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            lft_elbow = [xyz_lndmrk[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,xyz_lndmrk[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            rgt_elbow = [xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            
            lft_wrist = [xyz_lndmrk[mp_pose.PoseLandmark.LEFT_WRIST.value].x,xyz_lndmrk[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            rgt_wrist = [xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            lft_hip = [xyz_lndmrk[mp_pose.PoseLandmark.LEFT_HIP.value].x,xyz_lndmrk[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            rgt_hip = [xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_HIP.value].x,xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            lft_knee = [xyz_lndmrk[mp_pose.PoseLandmark.LEFT_KNEE.value].x,xyz_lndmrk[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            rgt_knee = [xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            
            lft_ankl = [xyz_lndmrk[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,xyz_lndmrk[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            rgt_ankl = [xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,xyz_lndmrk[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y] 
            
            #función para calcular el angulo entre punto inicial, medio y final
            def calc_th(a,b,c):
                  a = np.array(a)
                  b = np.array(b)
                  c = np.array(c)
                
                  th_rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                  th_deg = th_rad*180.0/np.pi
                  
                  if th_deg < 0:
                      th_deg += 360.0
                  return th_deg #si el angulo es menor a cero se agregan 360 para obtener el positivo
            
            # angulos de hombros, codos, cadera y rodillas (izq y derecha)
            lft_th_shldr = calc_th(lft_elbow, lft_shldr, lft_hip)
            rgt_th_shldr = calc_th(rgt_elbow, rgt_shldr, rgt_hip)
            
            lft_th_elbow = calc_th(lft_shldr, lft_elbow, lft_wrist)
            rgt_th_elbow = calc_th(rgt_shldr, rgt_elbow, rgt_wrist)
            
            lft_th_hip = calc_th(lft_shldr, lft_hip, lft_knee)
            rgt_th_hip = calc_th(rgt_shldr, rgt_hip, rgt_knee)
            
            lft_th_knee = calc_th(lft_hip, lft_knee, lft_ankl)
            rgt_th_knee = calc_th(rgt_hip, lft_knee, lft_ankl)
            
            
            print(xyz_lndmrk)
            print(lft_th_shldr)
            print(rgt_th_shldr)
            print(lft_th_elbow)
            print(rgt_th_elbow)
            print(lft_th_hip)
            print(rgt_th_hip)
            print(lft_th_knee)
            print(rgt_th_knee)
                   
        except:
            pass #si no se tiene estimaciones de una posición, pasa a otra
      
       
        
        #se renderiza detección
        mp_draw.draw_landmarks(img,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)#dibuja detecciones/estimaciones de la img y las conexiones
        
        cv2.imshow('captura de pos', img) #arroja img
    
        key = cv2.waitKey(1) #espera hasta que se presiona la tecla
        if key == ord("s"): #se para la captura al presionar s
            break
        

    capcam.release()
    cv2.destroyAllWindows() 
        
    
    

