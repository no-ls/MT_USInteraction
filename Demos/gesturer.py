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
    - downsampling helps, when you don't upsample before the optical flow
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
DEMO:
    - draw boat, that gets carried by the flow
    - -> move with flow vector at pixel
"""

class Gesturer(Demo):
    def __init__(self) -> None:
        super().__init__()
        self.radius = 7

        self.prev = []
        self.point = (0, 0)
        self.reset_counter = 0

    def init_point(self, masked:MatLike):
        if self.point == (0,0):
            h, w = masked.shape
            self.point = (int(w/2), int(h/2))

    def reset(self):
        print("reset point")
        self.point = (0,0)

    def do(self, frame:MatLike, masked:MatLike) -> MatLike:
        # super().do(frame, masked)

        small = cv2.pyrDown(masked)
        src = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        self.init_point(src)

        src = cv2.medianBlur(src,7) 
        op = cv2.adaptiveThreshold(src,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)

        kernel = np.ones((5,5),np.uint8)
        op = cv2.morphologyEx(op, cv2.MORPH_CLOSE, kernel)

        h, w = op.shape
        blank = np.zeros((h,w,3), np.uint8)

        # ===== TRACKING ===== # 

        op, flow = self.dense_OF(op)
        if len(flow) == 1: return src

        # == INTERACTION == #

        out = small
        # out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        self.animate_motion(out, flow)

        # == VIEWING == 

        if self.is_debug: # show flow
            # out = op # cv frame
            # out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            self.draw_flow(out, flow)

        out = cv2.pyrUp(out)
       # super().do(src, src)
        return out
    
    def animate_motion(self, frame:MatLike, flow):
        """Kinda, like this, but this not that good"""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        # get one motion vector and move from it to the next one it's pointing to
        # start in center of image and move with circles
        x, y = self.point
        
        # reset
        if x == 0 and y == 0:
            x = 100 # int(w/2)
            y = 240 # int(h/2)

        # move point using optical flow vectors
        point_mag = mag[y, x]

        if self.reset_counter >= 20:
            self.reset_counter = 0
            self.reset()
            new_x, new_y = self.point

        # skip if mag too low
        elif point_mag >= 0.5:

            # get 4 neighboring points around center
            offset = int(self.radius/2)
            top_fx, top_fy = flow[y+offset, x]
            left_fx, left_fy = flow[y, x-offset]
            bot_fx, bot_fy = flow[y-offset, x]
            right_fx, right_fy = flow[y,x+offset]  
            point_fx, point_fy = flow[y, x] # center
            
            # and average for smoother movement
            avg_fx = np.average([top_fx, left_fx, bot_fx, right_fx, point_fx])
            avg_fy = np.average([top_fy, left_fy, bot_fy, right_fy, point_fy])

            fac   = 1.4
            new_x = (int(x + avg_fx * fac))
            new_y = (int(y + avg_fy * fac))
            
            self.reset_counter = 0
        else:
            new_x = x
            new_y = y
            self.reset_counter += 1

        # DRAW
        cv2.circle(frame, (new_x, new_y), self.radius, COLORS.RED, -1)
        # self.write_text(frame, f"{int(new_x),int(new_y)}", (int(new_x),int(new_y)) )
        self.point = (new_x, new_y)

    # deprecated?
    def handle_oob(self, frame:MatLike):
        x, y = self.point
        
        # find edge of field
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt) 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        return frame

    # --------------------------
    
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
        self.prev = next # ?? -> damit geht's besser
        return frame, flow

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
# ----- MAIN ----- #
# video = "../Data/waterplay.mp4"
video = "../Data/new_swirl3-yadif2.mp4"
# video = "../Data/swirl.mp4"
player = Player(Gesturer(), video)
player.start_player()

