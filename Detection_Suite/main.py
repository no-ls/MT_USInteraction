import cv2
from cv2.typing import MatLike
import argparse
import time

from Default_vars import COLORS, KEYS
from Detection import Algorithm, ALGORITHMS, DEFAULT_A_VALUES

WINDOW = "Detection Suit"
DEFAULT_IMG = "../Data/Agar1.png"
DEFAULT_VID = "../Data/Lego.mp4"

COLOR_MAX = 255
US_AREA_THRESHOLD = 30
FPS_POS = (10, 20) 

class CL_Parser():
    """
    Read and parse the command line parameters.
    See: https://docs.python.org/3/library/argparse.html
    """

    def __init__(self) -> None:
        self.path = DEFAULT_IMG
        self.parser = argparse.ArgumentParser(
            prog="US_Detection_Suite",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Suite of different CV detection algorithms for Ultrasound Images",
            # epilog="""TODO"""
        )

        self.parser.add_argument('-src', '--source', type=str,
            default=DEFAULT_IMG, required=False, action="store",
            help="""The path to the image/video-file you want to use"""
        )

        # TODO pass debug as a cli arg

        self.parse_cmd_input()

    def parse_cmd_input(self):
        args = self.parser.parse_args()
        if args.source:
            self.path = args.source  
        print(f"[INFO] --> source: {self.path}")

class US_Area():
    """The area of the image/video that contains only the ultrasound scan"""
    def __init__(self, x, y, w, h) -> None:
        self.x = x # top right
        self.y = y # top right
        self.w = w
        self.h = h 

class Coordinator():
    """Coordinate between the different algorithms and settings"""

    def __init__(self, debug=False) -> None:
        self.debug = debug
        self.algorithm = ALGORITHMS[KEYS.ONE] # default init
        self.value = DEFAULT_A_VALUES[KEYS.ONE]

    def manage(self, src:MatLike) -> MatLike:
        """Manage all processing that is happening to an image and return it"""
        img, gray = self.preprocess_image(src)

        result = self.apply_algorithm(self.algorithm, self.value, img, gray)

        out = self.split_screen(self.debug, img, result)
        return out

    def preprocess_image(self, src:MatLike) -> tuple[MatLike]:
        """returns the preprocessed image as both BGR and Grayscale"""
        src = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        img = self.find_US_area(src, gray)
        return (img, gray)
    
    def split_screen(self, show:bool, src:MatLike, out:MatLike) -> MatLike:
        """Create a split screen from two frames to show them side by side"""
        if show:
            if(len(out.shape) < 3):
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            return cv2.hconcat([src, out])
        return src

    def apply_algorithm(self, algorithm: Algorithm, value, img, gray):
        return algorithm.apply(value, img, gray)

    def compare_algorithms(self):
        # get the 2 chosen algorithms and compare their output (Split Screen)
        pass

    def change_value(self, key):
        # TODO: make sure it can't overflow
        if key == KEYS.UP_ARROW:
            self.value += 1
        else:
            self.value -= 1

    def change_algorithm(self, key):
        """Switch between the implemented algorithms using the number keys 0-9 (not numpad)"""
        try:
            self.algorithm = ALGORITHMS[key]
            self.value = DEFAULT_A_VALUES[key]
        except:
            print("[ERR] Nothing algorithm here yet")
    
    # ----- HELPERS -----

    def find_US_area(self, src:MatLike, gray:MatLike) -> MatLike:
        """Find the original image so only the Ultrasound scan area is left"""
        
        _, thresh = cv2.threshold(gray, US_AREA_THRESHOLD, 255, cv2.THRESH_BINARY)
        eroded = cv2.erode(thresh, None, iterations=2) # get rid of text/lines
        contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        if self.debug:
            cv2.rectangle(src, (x, y), (x + w, y + h), COLORS.GREEN, 1)            

        self.area = US_Area(x, y, w, h)
        return src


class Viewer():
    """Handle all CV2 functions, that affect viewing the data"""

    def __init__(self, debug:bool=False) -> None:
        self.debug = debug
        self.parser = CL_Parser()
        self.coordinator = Coordinator(self.debug)
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.load_data()

    def load_data(self):
        if ".mp4" in self.parser.path:
            cap = cv2.VideoCapture(self.parser.path)
            self.show_video(cap)
        else:
            frame= cv2.imread(self.parser.path)
            self.show_img(frame)
        # evtl. open with mouse after running program (-> from tkinter import filedialog)

    def show_img(self, frame:MatLike) -> None:
        while True:
            out = self.coordinator.manage(frame)
            key = cv2.waitKeyEx(1)
            if key == ord("q") or key == KEYS.ESC:
                break
            elif key == ord("s"):
                self.save(out)
            elif key == KEYS.UP_ARROW or key == KEYS.DOWN_ARROW:
                self.coordinator.change_value(key)
            elif key >= KEYS.ZERO and key <= KEYS.NINE:
                self.coordinator.change_algorithm(key)
            cv2.imshow(WINDOW, out) 
        cv2.destroyAllWindows()

    def show_video(self, cap:cv2.VideoCapture) -> None:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            out = self.coordinator.manage(frame)
            self.write_fps(out)

            key = cv2.waitKeyEx(25)
            # if key != -1: print(key)
            if key == ord("q") or key == KEYS.ESC:
                self.stop(cap)
            elif key == ord("s"):
                self.save(out)
            elif key == KEYS.SPACE:
                self.pause()
            elif key == KEYS.UP_ARROW or key == KEYS.DOWN_ARROW:
                self.coordinator.change_value(key)
            elif key >= KEYS.ZERO and key <= KEYS.NINE:
                self.coordinator.change_algorithm(key)
            
            cv2.imshow('Video Playback', out)

    def save(self, img:MatLike):
        try:
            cv2.imwrite(f"../Data/Out/out_{time.time()}.png", img)
            print("saved image")
        except:
            print("[ERR] couldn't save")
    
    def stop(self, cap:cv2.VideoCapture):
        cv2.destroyAllWindows()
        cap.release()

    def pause(self):
        cv2.waitKey()

    def write_fps(self, img):
        # via: https://learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/
        self.new_frame_time = time.time()
        try:
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            fps = int(fps)
        except:
            fps = "x"
        self.prev_frame_time = self.new_frame_time
        text = f"fps: {str(fps)}"
        cv2.putText(img, text, FPS_POS, cv2.FONT_HERSHEY_PLAIN, 1, COLORS.WHITE, 1, cv2.LINE_AA)    

# ----- MAIN ----- #
def main():
    # TODO: put detection algorithms on number keys for quick change between
    viewer = Viewer(True)
    # coordinator = Coordinator(True)
    # coordinator.preprocess_image(viewer.frame)
    
main()
