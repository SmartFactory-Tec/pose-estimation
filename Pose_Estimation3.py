
import cv2
import time
import PoseModule as pm
import numpy as np
from math import acos, degrees

cap = cv2.VideoCapture('PoseVideos/run1.mp4')
pTime = 0
detector = pm.poseDetector()

brazada_delante = False
brazada_detras = False

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        x1 = lmList[14][1]  #codo derecho
        y1 = lmList[14][2]  #codo derecho
        x2 = lmList[12][1]  #hombro derecho
        y2 = lmList[12][2]  #hombro derecho
        x3 = lmList[16][1]  #muñeca derecha
        y3 = lmList[16][2]  #muñeca derecha

        # Cálculo de posición de brazo derecho
        p1 = np.array([x2, y2])
        p2 = np.array([x1, y1])
        p3 = np.array([x3, y3])

        l1 = np.linalg.norm(p2 - p3)    #lados de triangulo
        l2 = np.linalg.norm(p1 - p3)
        l3 = np.linalg.norm(p1 - p2)
        # Cálculo de ángulo
        angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))

        if angle > 75:
            print("brazada hacia atrás")
        if angle <70:
            print("brazada hacia delante")
        #visualización
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 6, (0, 255, 255), cv2.FILLED) #codo derecho
        cv2.circle(img, (lmList[12][1], lmList[12][2]), 6, (128, 0, 250), cv2.FILLED) #hombro derecho
        cv2.circle(img, (lmList[16][1], lmList[16][2]), 6, (255, 191, 0), cv2.FILLED) #muñeca derecha
        cv2.putText(img, str(int(angle)), (x2 + 30, y2), 1, 1.5, (128, 0, 250), 2)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3) #framerate

    cv2.imshow("Image", img)
    cv2.waitKey(1)