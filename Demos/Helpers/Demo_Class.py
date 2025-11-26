import cv2
import time
from .Parameters import COLORS
from abc import abstractmethod
from cv2.typing import MatLike

FPS_POS = (10, 20)

class Demo():
    def __init__(self) -> None:
        print("[DEMO] -", self.get_name())
        self.is_debug = False
        
        self.slider_val = 0
        self.slider_name = "Input" # What does the slider control?
        self.slider_contours = None

    def get_name(self)-> str:
        return self.__class__.__name__
    
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
        self.slider_val = input

    def visualize_slider_change(self, frame):
        if self.slider_contours != None:
            cv2.drawContours(frame, self.slider_contours, -1, COLORS.PURPLE, 1)
  
    # ----- ALGORITHM - STUFF ----- #

    def pre_tasks(self, frame:MatLike):
        """Bundled tasks before every demo"""
        self.show_fps(frame)

        if self.is_debug:
            self.visualize_slider_change(frame)

    @abstractmethod
    def do(self, frame:MatLike, gray:MatLike)-> MatLike:
        """Abstract Method to be overridden by the specific demos.
           This is where the algorithm gets executed. Returns an image"""
        self.pre_tasks(frame)
