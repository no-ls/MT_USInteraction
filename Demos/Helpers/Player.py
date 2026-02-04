import cv2
import numpy as np
from matplotlib import pyplot
from cv2.typing import MatLike
from .Parameters import KEYS, COLORS
from .Demo_Class import Demo
# from threading import Thread

VIDEO_ID = 2 # video id for the virtual camera
DEFAULT_VID = "../Data/Finger1.mp4"
DEFAULT_OUT_DIR = "../Data/Out"

WINDOW = "Demos"
TRACKBAR = "Input"
TRACKBAR_MIN = 0
TRACKBAR_MAX = 255
US_AREA_THRESHOLD = 40 # pong needs 20 / painter needs 40 (now for some reason)
PROBE_ARTIFACT = 10

# ----- HELPERS ----- #

class US_Area():
    """The area of the image/video that contains only the ultrasound scan"""
    def __init__(self) -> None:
        self.x = 0 # top right
        self.y = 0 # top right
        self.w = 0
        self.h = 0

    def update_US_area(self, gray:MatLike, threshold=US_AREA_THRESHOLD) -> bool:
        """Find the original image so only the Ultrasound scan area is left""" 
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        eroded = cv2.erode(thresh, None, iterations=1) # get rid of text/lines
        contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        
        # only change if the x/y coordinates change
        # NOTE: might have to change to w/h to actually detect zoom changes
        if self.x != x and self.y != y:
            self.set_area_parameters(x, y, w, h)
            return True
        return False

    def set_area_parameters(self, x:int, y:int, w:int, h:int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

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
        self.us_area_threshold = US_AREA_THRESHOLD

        self.cap = None

    def start_player(self):
        self.cap = cv2.VideoCapture(VIDEO_ID)
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not self.cap.isOpened():
            if ".mp4" in self.video:
                print("[INFO] - could not find a stream, will use video")
                self.cap = cv2.VideoCapture(self.video)
                self.play_video(self.cap)
            else:
                print("[INFO] - found image")
                frame = cv2.imread(self.video)
                self.show_img(frame)
        else:
            self.play_video(self.cap)
    
    def show_img(self, frame:MatLike)-> None:
        cv2.namedWindow(WINDOW)
        cv2.createTrackbar(self.demo.get_slider_name(), WINDOW, self.demo.slider_value,
                            self.demo.slider_max, self.demo.set_slider_input)
        
        frame, prepped = self.prepare_video(frame)
        while True:
            out = self.do_demo(frame, prepped)

            # KEYBOARD interactions
            key = cv2.waitKeyEx(25)
            if key == ord("q") or key == KEYS.ESC:
                break
            if key == ord("s"):
                self.save_frame(out)
            if key == ord("d"):
                self.demo.toggle_debug()
                pass
            cv2.imshow(WINDOW, out) 
        cv2.destroyAllWindows()

    def play_video(self, cap:cv2.VideoCapture):

        cv2.namedWindow(WINDOW)
        cv2.createTrackbar("Area", WINDOW, self.demo.us_area_threshold, 255, self.demo.adjust_area)
        cv2.createTrackbar(self.demo.get_slider_name(), WINDOW, self.demo.slider_value,
                            self.demo.slider_max, self.demo.set_slider_input)
        
        last_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # DEMO output
            frame, masked = self.prepare_video(frame) # only find us area once ?
            out = self.do_demo(frame, masked)

            # KEYBOARD interactions
            key = cv2.waitKeyEx(25)
            if key == ord("q") or key == KEYS.ESC:
                last_frame = out
                return
            else:
                self.handle_key_interaction(key, out)

            # OUT
            cv2.imshow(WINDOW, out)
        
        self.demo.on_finished(out)
        cv2.destroyAllWindows()
        cap.release()

    def handle_key_interaction(self, key:int, frame:MatLike):
        if key == ord("s"):
            self.save_frame(frame)
        if key == ord("d"):
            self.demo.toggle_debug()
        if key == KEYS.DOWN_ARROW or key == KEYS.UP_ARROW:
            self.demo.set_slider_with_keys(key)
            cv2.setTrackbarPos(self.demo.get_slider_name(), WINDOW, self.demo.slider_value)
        if key == ord("r"):
            self.demo.reset()
        elif key == ord(' '): # pause
            key = cv2.waitKey()

    # TODO show info about possible key interactions

    def prepare_video(self, frame:MatLike) -> tuple[MatLike]:
        """Rotate the video and find the US area"""
        src = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        changed_area = self.area.update_US_area(gray, self.demo.us_area_threshold)
        if changed_area: 
            self.demo.set_US_area(self.area) # ?
        roi = self.area.mask_US_area(src)
        return (src, roi)
    
    def do_demo(self, frame:MatLike, gray:MatLike):
        return self.demo.do(frame, gray)
    
    def save_frame(self, frame:MatLike):
        self.demo.save(frame)