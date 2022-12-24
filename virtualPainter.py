import cv2
import mediapipe as mp
import numpy as np

import HandTrackingModule as htm


class Painter:
    def __init__(self, img, color, prev, canvas):
        self.index_finger_x, self.index_finger_y = 0, 0
        self.two_finger_up = False
        self.colorRGB = color
        self.img = img
        self.xp, self.yp = prev
        self.canvas = canvas

    def addHeader(self):
        cv2.rectangle(self.img, (0, 0), (100, 100), (200, 100, 100), cv2.FILLED)
        cv2.rectangle(self.img, (100, 0), (200, 100), (100, 200, 100), cv2.FILLED)
        cv2.rectangle(self.img, (200, 0), (300, 100), (100, 100, 200), cv2.FILLED)

    def changeColor(self):
        if 0 < self.index_finger_x < 100:
            self.colorRGB = (200, 100, 100)
        elif 100 < self.index_finger_x < 200:
            self.colorRGB = (100, 200, 100)
        elif 200 < self.index_finger_x < 300:
            self.colorRGB = (100, 100, 200)

    def show(self):

        self.addHeader()
        detector = htm.handDetector()
        self.img = detector.findHands(self.img)
        lmList = detector.findPosition(self.img, draw=False)

        if len(lmList) != 0:
            self.index_finger_x, self.index_finger_y = lmList[8][1], lmList[8][2]

            if self.index_finger_y < 100:
                self.changeColor()

            cv2.circle(self.img, (self.index_finger_x, self.index_finger_y), 5, self.colorRGB, cv2.FILLED)
            cv2.line(self.img, (self.xp, self.yp), (self.index_finger_x, self.index_finger_y), self.colorRGB, 2)
            cv2.line(self.canvas, (self.xp, self.yp), (self.index_finger_x, self.index_finger_y), self.colorRGB, 2)

            self.xp = self.index_finger_x
            self.yp = self.index_finger_y

        return self.img, (self.xp, self.yp), self.canvas


cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 640)
canvas = np.zeros((480, 640, 3), np.uint8)
color = (0, 0, 0)
prev = (0,0)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    painter = Painter(img=img, color=color, prev=prev, canvas=canvas)
    img, prev, canvas = painter.show()
    color = painter.colorRGB
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(1)
