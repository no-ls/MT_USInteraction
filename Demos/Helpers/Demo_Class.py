import cv2
import time
from .Parameters import COLORS
from abc import abstractmethod
from cv2.typing import MatLike

FPS_POS = (10, 20)

class Demo():
    def __init__(self) -> None:
        pass

    def get_name(self):
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

    
    @abstractmethod
    def do(self, frame:MatLike, gray:MatLike)-> MatLike:
        """Abstract Method to be overridden by the specific demos."""
        pass
