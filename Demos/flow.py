import cv2
import numpy as np
from Helpers.Demo_Class import Demo, grayscale
from Helpers.Player import Player
from Helpers.Parameters import COLORS
from cv2.typing import MatLike


"""
Main implementation of the OPTICAL FLOW Demo:
    - visualization of water movement
    - ball falls is moved across the screen following the water flow found in the ultrasound image
CONTROLS:
    - r = reset the ball to its original position

Ultrasound settings:
    - basically, brighten the entire image similarly throughout
    - might require reducing amplification of the near and far field
    - Example /w Philips SDR-1200:
        - Amplification of entire field: 60 (out of 60)
        - Amplification of near field: 00
        - Amplification of the far field: 0.0
        - Focus: Middle

NOTE on Framerate
    - as ultrasound images are recorded sequentially a low framerate of the ultrasound machine may cause images to only update section wise
    - try and maximize the framerate of the ultrasound machine, typically using only one focus point should work 
        
"""

INIT_X = 100
INIT_Y = 240

MIN_MAGNITUDE = 0.5 
MAX_COUNTER = 20
RADIUS = 7

class Gesturer(Demo):
    def __init__(self) -> None:
        super().__init__()
        self.radius = RADIUS

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

        return out
    
    def animate_motion(self, frame:MatLike, flow):
        """Kinda, like this, but this not that good"""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        # get one motion vector and move from it to the next one it's pointing to
        # start in center of image and move with circles
        x, y = self.point
        
        # reset (to a good point in the flow -> middle of image might not have a lot of movement)
        if x == 0 and y == 0:
            x = INIT_X # int(w/2)
            y = INIT_Y # int(h/2)

        # move point using optical flow vectors
        point_mag = mag[y, x]

        # autoreset if stops moving for too long
        if self.reset_counter >= MAX_COUNTER:
            self.reset_counter = 0
            self.reset()
            new_x, new_y = self.point

        # skip if mag too low
        elif point_mag >= MIN_MAGNITUDE:

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
        self.prev = next # update frame
        return frame, flow

    def draw_flow(self, frame:MatLike, flow):
        """Draw the optical flow direction vectors (via: ChatGPT)"""
        step = 10 # draw every 10 pixels (avoid clutter)
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

video = "../Data/new_swirl3.mp4"
player = Player(Gesturer(), video)
player.start_player()

