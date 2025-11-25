import cv2
import numpy as np
from Helpers.Demo_Class import Demo
from Helpers.Player import Player
from Helpers.Parameters import COLORS
from cv2.typing import MatLike

"""
Use optical flow to differentiate between movement, aka gestures

LOG:
Tutorial: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
    - finding brightest spot works, but not very consistently -> keeps changing
    - using FlowFarnback for dense optical flow works, but has similar issues
        -> tracking stays around an area, and doesn't move like the bubbles do
        -> Richtung lässt sich aber erahnen
    - TODO try preprocessing
"""

class Flower(Demo):
    def __init__(self) -> None:
        super().__init__()
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
        self.prev = []
        self.color = np.random.randint(0, 255, (100, 3))

    def do(self, frame: MatLike, gray_cut: MatLike) -> MatLike:
        
        lk = False

        if not lk:
        # do OPTICAL FLOW (full)
            if len(self.prev) == 0:
                frame = cv2.cvtColor(gray_cut, cv2.COLOR_GRAY2BGR)
                self.prev = gray_cut
                self.hsv = np.zeros_like(frame)
                self.hsv[..., 1] = 255
                return frame
            next = gray_cut
            flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            self.hsv[..., 0] = ang*180/np.pi/2
            self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            out = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
            out = cv2.add(frame, out)

        # do OPTICAL FLOW (LK)
        else:
            frame = cv2.cvtColor(gray_cut, cv2.COLOR_GRAY2BGR)

            if len(self.prev) == 0:
                self.prev = gray_cut
                # TODO better detect a bubble to track
                brightest = brighest_spot(50, frame, gray_cut) # detect the brightest point aka bubble
                self.p0 =  np.array([brightest], dtype=np.float32) # to proper datatype
                # self.p0 = cv2.goodFeaturesToTrack(self.prev, mask=None, **self.feature_params)
                self.mask = np.zeros_like(self.prev)
                return frame
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev, gray_cut, self.p0, None, **self.lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[st==1]
                good_old = self.p0[st==1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2) # lines
                out = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)

        # out = frame
        self.show_fps(out)
        return out


def brighest_spot(value:int, img:MatLike, gray:MatLike):
    img = cv2.GaussianBlur(img, (7,7), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    # print("rect", cv2.boundingRect([maxLoc]))
    # cv2.circle(img, maxLoc, 5, COLORS.BLUE, 2)
    # cv2.putText(img, " brightest", maxLoc, 1, 1, COLORS.BLUE, 1)
    x, y = maxLoc
    return [[x, y]]

# ----- MAIN ----- #
video = "../Data/bubbles.mp4"
player = Player(Flower(), video)
player.start_player()

