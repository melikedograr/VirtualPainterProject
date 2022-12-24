import cv2
import mediapipe as mp
import numpy as np
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 640)
canvas = np.zeros((480, 640, 3), np.uint8)
color = (0, 0, 0)
xp, yp = (0, 0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #### add header
    cv2.rectangle(img, (0, 0), (100, 100), (200, 100, 100), cv2.FILLED)
    cv2.rectangle(img, (100, 0), (200, 100), (100, 200, 100), cv2.FILLED)
    cv2.rectangle(img, (200, 0), (300, 100), (100, 100, 200), cv2.FILLED)
    cv2.rectangle(img, (300, 0), (400, 100), (0, 0, 0), cv2.FILLED)
    ##########

    detector = htm.handDetector()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        index_finger_x, index_finger_y = lmList[8][1], lmList[8][2]

        ## change color
        if index_finger_y < 100:
            if 0 < index_finger_x < 100:
                color = (200, 100, 100)
            elif 100 < index_finger_x < 200:
                color = (100, 200, 100)
            elif 200 < index_finger_x < 300:
                color = (100, 100, 200)
            elif 300 < index_finger_x < 400:
                color = (0,0,0)
        ###########

        #### draw

        if xp == 0 and yp == 0:
            xp = index_finger_x
            yp = index_finger_y

        cv2.circle(img, (index_finger_x, index_finger_y), 5, color, cv2.FILLED)
        cv2.line(img, (xp, yp), (index_finger_x, index_finger_y), color, 2)
        cv2.line(canvas, (xp, yp), (index_finger_x, index_finger_y), color, 2)

        xp = index_finger_x
        yp = index_finger_y
        ######

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    final = cv2.bitwise_and(img, imgInv)
    final = cv2.bitwise_or(final, canvas)

    cv2.imshow("Image", final)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(1)
