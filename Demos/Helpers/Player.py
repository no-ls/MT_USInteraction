from abc import abstractmethod
import cv2
from cv2.typing import MatLike
from .Parameters import KEYS
from .Demo_Class import Demo

VIDEO_ID = 2 # video id for the virtual camera
DEFAULT_VID = "../Data/Finger1.mp4"

WINDOW = "Painter"
US_AREA_THRESHOLD = 30

# ----- HELPERS ----- #

class US_Area():
    """The area of the image/video that contains only the ultrasound scan"""
    def __init__(self, x, y, w, h) -> None:
        self.x = x # top right
        self.y = y # top right
        self.w = w
        self.h = h 

def find_US_area(gray:MatLike) -> MatLike:
    """Find the original image so only the Ultrasound scan area is left"""
    
    _, thresh = cv2.threshold(gray, US_AREA_THRESHOLD, 255, cv2.THRESH_BINARY)
    eroded = cv2.erode(thresh, None, iterations=2) # get rid of text/lines
    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    return US_Area(x, y, w, h)


# ----- PLAYER ----- # 

class Player():
    def __init__(self, demo:Demo) -> None:
        self.demo = demo
        self.area = None

    def start_player(self):
        cap = self.load_input()
        self.play_video(cap)

    def load_input(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(VIDEO_ID)
        if not cap.isOpened():
            print("[INFO] - could not find a stream, will use video")
            cap = cv2.VideoCapture(DEFAULT_VID)
        return cap
    
    def play_video(self, cap:cv2.VideoCapture):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKeyEx(25)
            if key == ord("q") or key == KEYS.ESC:
                break

            frame, gray = self.prepare_video(frame)
            out = self.do_demo(frame, gray)

            cv2.imshow(WINDOW, out)
        
        cv2.destroyAllWindows()
        cap.release()

    def prepare_video(self, frame:MatLike) -> tuple[MatLike]:
        """Rotate the video and find the US area"""
        src = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        self.area = find_US_area(gray)
        return (src, gray)
    
    def do_demo(self, frame:MatLike, gray:MatLike):
        return self.demo.do(frame, gray)