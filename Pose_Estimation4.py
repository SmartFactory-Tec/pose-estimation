
import cv2
import time
import PoseModule as pm
import numpy as np
from math import acos, degrees

cap = cv2.VideoCapture('PoseVideos/Sitting_down.mp4')
pTime = 0
detector = pm.poseDetector()

brazada_delante = False
brazada_detras = False

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        #Pierna Izquierda
        x1 = lmList[23][1]  #cadera derecho
        y1 = lmList[23][2]  #cadera derecho
        x2 = lmList[25][1]  #rodilla derecho
        y2 = lmList[25][2]  #rodilla derecho
        x3 = lmList[27][1]  #tobillo derecho
        y3 = lmList[27][2]  #tobillo derecho

        #Pierna Derecha
        x4 = lmList[24][1]
        y4 = lmList[24][2]
        x5 = lmList[26][1]
        y5 = lmList[26][2]
        x6 = lmList[28][1]
        y6 = lmList[28][2]

        # Cálculo de posición de pierna izquierda
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([x3, y3])

        l1 = np.linalg.norm(p2 - p3)    #lados de triangulo
        l2 = np.linalg.norm(p1 - p3)
        l3 = np.linalg.norm(p1 - p2)
        # Cálculo de ángulo pierna izquierda
        angle1 = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))

        # Cálculo de posición de pierna derecha
        p4 = np.array([x4, y4])
        p5 = np.array([x5, y5])
        p6 = np.array([x6, y6])

        l4 = np.linalg.norm(p5 - p6)  # lados de triangulo
        l5 = np.linalg.norm(p4 - p6)
        l6 = np.linalg.norm(p4 - p5)
        # Cálculo de ángulo pierna izquierda
        angle2 = degrees(acos((l4 ** 2 + l6 ** 2 - l5 ** 2) / (2 * l4 * l6)))

        if (angle1 and angle2) > 120:
            print("de pie")
        if (angle1 and angle2) < 120:
            print("sentado")
        #visualización
        #cv2.circle(img, (lmList[23][1], lmList[23][2]), 6, (0, 255, 255), cv2.FILLED) #codo derecho
        #cv2.circle(img, (lmList[25][1], lmList[25][2]), 6, (128, 0, 250), cv2.FILLED) #hombro derecho
        #cv2.circle(img, (lmList[27][1], lmList[27][2]), 6, (255, 191, 0), cv2.FILLED) #muñeca derecha
        cv2.putText(img, str(int(angle1)), (x2 + 30, y2), 1, 1.5, (128, 0, 250), 2)
        cv2.putText(img, str(int(angle2)), (x5 + 30, y5), 1, 1.5, (128, 0, 250), 2)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3) #framerate

    cv2.imshow("Image", img)
    cv2.waitKey(1)