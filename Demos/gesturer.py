import cv2
import numpy as np
from Helpers.Demo_Class import Demo, grayscale
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
/W DENSE
    - bubbles not very reliable to find direction specifically angle
    - problem is that the magnitudes keeps changing (i.e biggest mag. not at the same spot)
    - no real motion is being detected (when drawing vectors), sometime half of a circular motion
    - an Ecken öfters "wirbel", manchmal auch in der Mitte, aber eher sporadisch
/W LK
    - init with single bubble -> pretty much stays in the same spot afterwards
    - init with color quantized image => only a little better
PROBLEMS:
    - Images too bad -> bubbles are not getting tracked properly
    - different brightness levels: higher = darker, lower = brighter -> tracking gets lost
    - Bubbles are not circular :( -> so prob. no hough circles
SUBTRACT BACKGROUND
    - bubbles are more visible
    - but does not seem to change much for optical flow
        - LK on contour -> does not find any good points
        - LK (normal) -> couldn't get it to work
        - Dense => looks very similar to the one without SB
==> Feature too small/too bad to work with regular optical flow
    - try to enhance features -> isolate + make "bigger"
        - rm background kinda works (but OF doesnt really get better)
        - morph operation don't do that much either
            - dilate => blasen werden zu groß -> Überlappungen
        - I DON'T KNOW, man...
    - try PIV Techniques
        - funktioniert auch nicht so viel besser
ADAPTIVE THRESHOLD
    - regular blurring does not help as much as downsampling + morph op does
    - kann keine Unterschied zw. Gaussian und Mean C sehen
    - better than regular one
    - even better when paired with downsampling (pyrdown) + morph op (closeing) + upsampling before optical flow
    - Bewegungen immer noch eher sporadisch,
    - OTSU -> just removes basically everything
is framerate the issue ?? -> CUDA benutzen
"""

"""
GET FLOW:
    - segment bubbles, then sparse optical flow
        -> only segment on first frame
    - dense optical flow on entire frame
GET GESTURE:
    - Richtungsvektoren von den größten Flächen kriegen, und dann daraus Form erstellen?
    - Richtungsvektoren von größter Fläche frame für frame aneinander pappen?
    - Track one "color" and record movement -> translate to shape
"""

# ??? use the contours to track instead of optical flow

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
        self.mags = []
        self.backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.backsub2 = cv2.createBackgroundSubtractorKNN(detectShadows=False)

        # self.slider_value = 130
        # self.slider_max = 255

    def do(self, frame:MatLike, masked:MatLike) -> MatLike:
        # super().do(frame, masked)

        src = cv2.pyrDown(masked)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        src = cv2.medianBlur(src,11) # does not do that much
        # src = cv2.bilateralFilter(src,9,75,75) # ?? does not do that much either
        # src = cv2.GaussianBlur(src,(5,5),0)
        op = cv2.adaptiveThreshold(src,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2) # does help with bg removal
        # ret2, op = cv2.threshold(src,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # nope
        # op = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        kernel = np.ones((5,5),np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        op = cv2.morphologyEx(op, cv2.MORPH_CLOSE, kernel)

        # op = cv2.pyrUp(op)

        h, w = op.shape
        blank = np.zeros((h,w,3), np.uint8)
        # blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)

        # TODO: try opencv Trackers instead (e.g. https://pyimagesearch.com/2018/07/30/opencv-object-tracking/, https://github.com/opencv/opencv_contrib/blob/master/modules/tracking/samples/tracker.py)

        # Do optical flow
        lk = True
        if not lk:
            op, flow = self.dense_OF(op)
            if len(flow) == 1: return src
        
            op = cv2.cvtColor(op, cv2.COLOR_GRAY2BGR)
            if self.is_debug: op = blank
            self.draw_flow(op, flow)
        else:
            contours, op = self.segment(op)
            src = self.LK_OF(op)
            
        # frame = cv2.pyrUp(frame)


        src = op
        src = cv2.pyrUp(src)
        # super().do(src, src)
        return src
    
    def dense_OF(self, frame:MatLike):
        """Calculate the dense optical flow of consecutive frames"""
        gray = grayscale(frame)
        if len(self.prev) == 0:
            self.prev = gray
            self.hsv = np.zeros_like(frame)
            self.hsv[..., 1] = 255
            return frame, [[[]]]
        next = gray
        flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # fx = flow[..., 0]
        # fy = flow[..., 1]
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # magnitude and direction
        # self.prev = next # ??? -> hat manchmal den Kreis drinn
        return frame, flow

    def LK_OF(self, frame:MatLike):
        """Calculate the Lucas Kanade to track some points (good features to track) with optical flow"""
        masked = grayscale(frame)

        if len(self.prev) == 0:
            self.prev = masked
            self.p0 = cv2.goodFeaturesToTrack(self.prev, mask=None, **self.feature_params)
            self.mask = np.zeros_like(self.prev)
            return masked
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev, masked, self.p0, None, **self.lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(frame, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2) # lines
            out = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
        return out

    def draw_flow(self, frame:MatLike, flow):
        """Draw the optical flow direction vectors (via: ChatGPT)"""
        step = 10 # 16  # draw every 16 pixels (avoid clutter)
        h, w, _ = frame.shape

        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]

                # Draw arrow: start -> end
                cv2.arrowedLine(
                    frame,
                    (x, y),
                    (int(x + fx), int(y + fy)),
                    color=(0, 255, 0),
                    thickness=1,
                    tipLength=0.3
                )

# --- TESTS ---- #

    def do_bg(self, frame:MatLike, masked:MatLike) -> MatLike:
        """DO background subtraction and then LK"""
        
        frame = cv2.GaussianBlur(frame, (5, 5), 1)
        fgmask = self.backsub2.apply(frame)

        # contours, result = self.segment(fgmask)

        contours, _ = cv2.findContours(fgmask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        _, gray = cv2.threshold(fgmask,100,255,cv2.THRESH_BINARY)
        # print(contours)
        out = gray

        if len(self.prev) == 0:
            self.prev = gray
            self.p0 = np.squeeze(contours[0])
            self.p0 = self.p0.astype(np.float32)
            # self.p0 = cv2.goodFeaturesToTrack(self.prev, mask=None, **self.feature_params) # does not find any
            self.mask = np.zeros_like(self.prev)
            # print(self.p0)
            return gray
        
        pts = np.squeeze(contours[0])
        # pts = cnt.reshape(-1, 1, 2).astype(np.float32)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.p0, pts, **self.lk_params)

        # # Select good points
        if p1 is not None:
            good_new = p1[st.flatten() ==1]
            good_old = self.p0[st.flatten() ==1]
            # NOTHING?

        # # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2) # lines
            out = cv2.circle(gray, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            

        return out

    def do_sub_dense(self, frame:MatLike, masked:MatLike) -> MatLike:
        """Do background subtraction and then dense optical flow"""

        # Background subtraction
        frame = cv2.GaussianBlur(frame, (5, 5), 1)
        
        fgmask = self.backsub2.apply(frame)
        _, gray = cv2.threshold(fgmask,100,255,cv2.THRESH_BINARY)

        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if len(self.prev) == 0:
            self.prev = gray
            self.hsv = np.zeros_like(frame)
            self.hsv[..., 1] = 255
            return fgmask
        next = gray
        flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # magnitude and direction

        max_idx = np.unravel_index(np.argmax(mag), mag.shape)
        y, x = max_idx
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

        self.hsv[..., 0] = ang*180/np.pi/2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        out = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        out = cv2.add(frame, out)
        
        return out
    
    def do_old2(self, frame: MatLike, masked: MatLike) -> cv2.Mat:

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

        # IDEA: try drawing biggest 10 mags -> where are they -> in a circle mayhaps

        # LOL: try draw line between mag centers
        # self.mags.append((x, y))
        # self.draw_magnitudes(frame)


        # Visualization OF
        self.hsv[..., 0] = ang*180/np.pi/2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        out = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        out = cv2.add(frame, out)

        self.show_fps(out)
        return out
    
    def draw_magnitudes(self, frame: MatLike):
        if len(self.mags) == 0:
            return

        for i in range (0, len(self.mags)-1):
            start = self.mags[i]
            end = self.mags[i+1]
            cv2.line(frame, start, end, COLORS.RED, 3)
 
    # both optical flows working
    def do_optical(self, frame: cv2.Mat, masked: cv2.Mat) -> cv2.Mat:
        
        lk = True
        # self.is_debug = True
        _, masked = self.segment(masked)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        # ??: LK auf contour

        # ret, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        if not lk:
        # do OPTICAL FLOW (full)
            if len(self.prev) == 0:
                # frame = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
                self.prev = gray
                self.hsv = np.zeros_like(masked)
                self.hsv[..., 1] = 255
                return masked
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
            cv2.circle(masked, (x, y), 4, (255, 255, 255), -1)

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
            cv2.line(masked, (x, y), end_point, (0, 0, 255), 3)
            cv2.circle(masked, (x, y), 4, (0, 255, 0), -1)

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
            out = cv2.add(masked, out)

        # do OPTICAL FLOW (LK)
        else:
            # frame = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
            masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

            if len(self.prev) == 0:
                self.prev = masked
                # TODO better detect a bubble to track
                # brightest = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
                brightest = brightest_spot(50, frame, masked) # detect the brightest point aka bubble
                self.p0 =  np.array([brightest], dtype=np.float32) # to proper datatype
                # self.p0 = cv2.goodFeaturesToTrack(self.prev, mask=None, **self.feature_params)
                self.mask = np.zeros_like(self.prev)
                return masked
            
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
                out = cv2.circle(masked, (int(a), int(b)), 5, self.color[i].tolist(), -1)

        # out = frame
        self.show_fps(out)
        return out

def brightest_spot(value:int, img:MatLike, gray:MatLike):
    img = cv2.GaussianBlur(img, (7,7), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    # print("rect", cv2.boundingRect([maxLoc]))
    # cv2.circle(img, maxLoc, 5, COLORS.BLUE, 2)
    # cv2.putText(img, " brightest", maxLoc, 1, 1, COLORS.BLUE, 1)
    x, y = maxLoc
    return [[x, y]]

# ----- MAIN ----- #
# video = "../Data/bubbles.mp4"
video = "../Data/swirl.mp4"
player = Player(Gesturer(), video)
player.start_player()

