import cv2
import time
import numpy as np
from .Parameters import COLORS
from abc import abstractmethod
from cv2.typing import MatLike

FPS_POS = (10, 20)

DEFAULT_SLIDER_VALUE = 130 # 6
DEFAULT_MAX_SLIDER_VALUE = 255 # 20
DEFAULT_SLIDER_TEXT = "Colors"

class Demo():
    def __init__(self) -> None:
        print("[DEMO] -", self.get_name())
        self.is_debug = False
        self.area_w = None
        self.area_h = None
        self.masked_w = None
        self.masked_h = None

        self.slider_value = DEFAULT_SLIDER_VALUE
        self.slider_max = DEFAULT_MAX_SLIDER_VALUE
        self.slider_name = DEFAULT_SLIDER_TEXT
        self.slider_contours = None

    def get_name(self)-> str:
        return self.__class__.__name__

    def write_text(self, frame:MatLike, text:str, pos:tuple[int])-> None:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_PLAIN, 1, COLORS.WHITE, 1, cv2.LINE_AA)    

    def show_fps(self, frame:MatLike)->MatLike:
        self.new_frame_time = time.time()
        try:
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            fps = int(fps)
        except:
            fps = "x"
        self.prev_frame_time = self.new_frame_time
        text = f"fps: {str(fps)}"
        cv2.putText(frame, text, FPS_POS, cv2.FONT_HERSHEY_PLAIN, 1, COLORS.WHITE, 1, cv2.LINE_AA)    

    def toggle_debug(self):
        self.is_debug = not self.is_debug
        print(f"[DEBUG] - {self.is_debug}")

    def get_slider_name(self)-> str:
        return self.slider_name
    
    def set_slider_input(self, input:int):
        self.slider_value = input

    def visualize_slider_change(self, frame):
        if self.slider_contours != None:
            cv2.drawContours(frame, self.slider_contours, -1, COLORS.PURPLE, 1)
  
    def set_dimensions(self, frame:MatLike, masked:MatLike):
        """Set the dimensions of the frame area"""
        if self.area_h == None and self.area_w == None:
            self.area_h, self.area_w, _ = frame.shape
            self.masked_h, self.masked_w, _ = masked.shape # ??
            
            print(f"[INFO] Frame is: {self.area_h} x {self.area_w} px (h x w)")
            print(f"[INFO] Mask  is: {self.masked_h} x {self.masked_w} px (h x w)") # ??

    # ----- ALGORITHM - STUFF ----- #

    def pre_tasks(self, frame:MatLike, masked:MatLike):
        """Bundled tasks before every demo"""
        self.show_fps(frame)
        self.set_dimensions(frame, masked)

    @abstractmethod
    def do(self, frame:MatLike, masked:MatLike)-> MatLike:
        """Abstract Method to be overridden by the specific demos.
           This is where the algorithm gets executed. Returns an image"""
        self.pre_tasks(frame, masked)

    @abstractmethod
    def segment(self, frame:MatLike)-> MatLike:
        """Default segmentation method. Returns the 'brightest' contours"""

        # blur
        frame = cv2.GaussianBlur(frame, (5,5), 0)

        contours, result = color_quantization(frame, self.slider_value)
        # contours, result = threshold(frame, self.slider_value)
        if self.is_debug:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(result, contours, -1, COLORS.PURPLE, 1)
            return contours, result
        
        return contours, frame
    
# ----- SEGMENTATION ----- # 

def grayscale(frame:MatLike)-> MatLike:
    """Convert to gray, if it is not already"""
    gray = frame
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def threshold(frame:MatLike, value)-> tuple[MatLike, MatLike]:
    gray = grayscale(frame)
    _, thresh = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, thresh

def color_quantization(frame:MatLike, num_colors) -> tuple[MatLike, MatLike]:
    """Use color quantization to segment the image. Returns found contours"""
    # via: https://stackoverflow.com/a/66339640

    # convert to gray as float in range 0 to 1
    gray = grayscale(frame)
    gray = gray.astype(np.float32)/255

    # quantize and convert back to range 0 to 255 as 8-bits
    result = 255*np.floor(gray*num_colors+0.5)/num_colors
    result = result.clip(0,255).astype(np.uint8)

    # find contours
    _, thresh = cv2.threshold(result, 75, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # TODO: filter out the biggest contour -> is the entire area (for some reason)
    return contours, result