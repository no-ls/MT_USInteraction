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
    -> Try with better data -> faster motion, more bubbles, etc.
"""

"""
GET FLOW:
    - segment bubbles, then sparse optical flow
    - dense optical flow on entire frame
GET GESTURE:
    - Richtungsvektoren von den größten Flächen kriegen, und dann daraus Form erstellen?
    - Richtungsvektoren von größter Fläche frame für frame aneinander pappen?
    - Track one "color" and record movement -> translate to shape
"""

class Gesturer(Demo):
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
    
    def do(self, frame: MatLike, masked: MatLike) -> cv2.Mat:

        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        if len(self.prev) == 0:
            # frame = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
            self.prev = gray
            self.hsv = np.zeros_like(frame)
            self.hsv[..., 1] = 255
            return frame
        next = gray
        flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        fy = flow[..., 1]
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # magnitude and direction

        # TODO

        # find center of biggest motion areas
        # use motion vectors to create a "gesture" (line)
        
        mag_idx = np.unravel_index(np.argmax(mag), mag.shape)
        y, x = mag_idx
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        vx, vy = flow[y, x]

        scale = 5
        end_point = (int(x + vx * scale), int(y + vy * scale))
        cv2.line(frame, (x, y), end_point, (0, 0, 255), 3)

        # Visualization OF
        self.hsv[..., 0] = ang*180/np.pi/2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        out = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        out = cv2.add(frame, out)

        self.show_fps(out)
        return out

    def do_test(self, frame: cv2.Mat, masked: cv2.Mat) -> cv2.Mat:
        
        lk = False
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        if not lk:
        # do OPTICAL FLOW (full)
            if len(self.prev) == 0:
                # frame = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
                self.prev = gray
                self.hsv = np.zeros_like(frame)
                self.hsv[..., 1] = 255
                return frame
            next = gray
            flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fx = flow[..., 0]
            fy = flow[..., 1]

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # magnitude and direction
            # magnitude is the distance that the pixel moved
            # angle is orientation where the pixel moved to
            # avg. mag = average moved distance
            # avg. ang = average movement angle

            mag_blur = cv2.GaussianBlur(mag, (25, 25), 0)
            max_mag = np.argmax(mag_blur)
            max_idx = np.unravel_index(np.argmax(mag), mag.shape)
            y, x = max_idx
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

            # -----

            # biggest area of magnitude/ avg. angle?
            # print("ang", ang.mean(), ang.max(), ang.min())
            # TODO: make a shape out of "all" motion vectors
            
            # ---

            # avg_vx = fx.mean()
            # avg_vy = fy.mean()
            # print(avg_vx, avg_vy)

            # ---

            # works (kinda)
            # find the max location and get flow vector
            max_idx = np.unravel_index(np.argmax(mag_blur), mag_blur.shape)
            y, x = max_idx
            vx, vy = flow[y, x]

            scale = 5
            end_point = (int(x + vx * scale), int(y + vy * scale))
            cv2.line(frame, (x, y), end_point, (0, 0, 255), 3)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # ----

            # get angle of vector with highest mag
            # max = np.max(mag)
            # max_idx = np.where(mag == max)
            # print("angle", ang[max_idx[0]])
            # # print("mag", np.max(mag), "index", np.where(mag == max))

            # max_idx = np.unravel_index(np.argmax(mag), mag.shape)
            # max_vector = flow[max_idx[0], max_idx[1], :]  # (vx, vy)
            # print("Max magnitude:", mag[max_idx])
            # print("Max flow vector:", max_vector)
            # print("Pixel location:", (max_idx[1], max_idx[0])) # strongest motion

            # map as line
            # fx, fy = flow[:, :, 0], flow[:, :, 1]
            # lines = np.vstack([fx, fy, np.ones(fx.shape)])
            # print("fx", fx, "fy", fy)
            # print(lines)


            # Visualization OF
            self.hsv[..., 0] = ang*180/np.pi/2
            self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            out = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
            out = cv2.add(frame, out)

        # do OPTICAL FLOW (LK)
        else:
            # frame = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
            masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

            if len(self.prev) == 0:
                self.prev = masked
                # TODO better detect a bubble to track
                brightest = brighest_spot(50, frame, masked) # detect the brightest point aka bubble
                self.p0 =  np.array([brightest], dtype=np.float32) # to proper datatype
                # self.p0 = cv2.goodFeaturesToTrack(self.prev, mask=None, **self.feature_params)
                self.mask = np.zeros_like(self.prev)
                return frame
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev, masked, self.p0, None, **self.lk_params)

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
player = Player(Gesturer(), video)
player.start_player()

