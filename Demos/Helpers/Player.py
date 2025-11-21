import os
import cv2
import time
import numpy as np
from abc import abstractmethod
from cv2.typing import MatLike
from .Parameters import KEYS, COLORS
from .Demo_Class import Demo

VIDEO_ID = 2 # video id for the virtual camera
DEFAULT_VID = "../Data/Finger1.mp4"
DEFAULT_OUT_DIR = "../Data/Out"

WINDOW = "Painter"
US_AREA_THRESHOLD = 20
PROBE_ARTIFACT = 10

# ----- HELPERS ----- #

class US_Area():
    """The area of the image/video that contains only the ultrasound scan"""
    def __init__(self) -> None:
        # self.x = x # top right
        # self.y = y # top right
        # self.w = w
        # self.h = h
        pass

    def find_US_area(self, gray:MatLike):
        """Find the original image so only the Ultrasound scan area is left""" 
        _, thresh = cv2.threshold(gray, US_AREA_THRESHOLD, 255, cv2.THRESH_BINARY)
        eroded = cv2.erode(thresh, None, iterations=2) # get rid of text/lines
        contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.x, self.y, self.w, self.h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    
    def mask_US_area(self, img):
        """Return a copy of the image that is masked to the US-Area (rest is black)"""
        black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        black = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(black, (self.x, self.y), (self.x+self.w, self.y+self.h), COLORS.WHITE, -1)
        cv2.rectangle(black, (self.x, self.y), (self.x+self.w, self.y+PROBE_ARTIFACT), COLORS.BLACK, -1)
        # -> remove the very top of the area, as there are a lot of probe artifacts and nothing else
        
        # HACK: remove bottom of area for testing
        cv2.rectangle(black, (self.x, self.y+self.h-PROBE_ARTIFACT), (self.x+self.w, self.h+self.h), COLORS.BLACK, -1)

        _, mask = cv2.threshold(black, 127, 255, 0)
        return cv2.bitwise_and(img, img, mask=mask)
        # via: https://stackoverflow.com/a/42007566
        # ROI = image[y:y+h, x:x+w] (https://stackoverflow.com/a/58177717) 


# ----- PLAYER ----- # 

class Player():
    def __init__(self, demo:Demo, video:str=DEFAULT_VID) -> None:
        self.demo = demo
        self.area = None
        self.video = video
        self.area = US_Area()

    def start_player(self):
        cap = self.load_input()
        self.play_video(cap)

    def load_input(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(VIDEO_ID)
        if not cap.isOpened():
            print("[INFO] - could not find a stream, will use video")
            cap = cv2.VideoCapture(self.video)
        return cap
    
    def play_video(self, cap:cv2.VideoCapture):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # DEMO output
            frame, prepped = self.prepare_video(frame)
            out = self.do_demo(frame, prepped)

            # KEYBOARD interactions
            key = cv2.waitKeyEx(25)
            if key == ord("q") or key == KEYS.ESC:
                break
            if key == ord("s"):
                self.save_frame(out)

            # OUT
            cv2.imshow(WINDOW, out)
        
        cv2.destroyAllWindows()
        cap.release()

    def prepare_video(self, frame:MatLike) -> tuple[MatLike]:
        """Rotate the video and find the US area"""
        src = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        self.area.find_US_area(gray)
        roi = self.area.mask_US_area(gray)
        return (src, roi)
    
    def do_demo(self, frame:MatLike, gray:MatLike):
        return self.demo.do(frame, gray)
    
    def save_frame(self, frame:MatLike):
        if not os.path.isdir(DEFAULT_OUT_DIR):
            os.makedirs(DEFAULT_OUT_DIR)

        try:
            filepath = f"{DEFAULT_OUT_DIR}/out-{self.demo.get_name()}_{time.time()}.png"
            cv2.imwrite(filepath, frame)
            print("[INFO] saved image @", filepath)
        except:
            print("[ERR] couldn't save")