import cv2
import numpy as np
from Helpers.Demo_Class import Demo
from Helpers.Player import Player
from Helpers.Parameters import COLORS
from cv2.typing import MatLike
import pyautogui
import mouse

"""
Main implementation of the DEFORM-able Demo:
    - use a deformable stress ball as an input device
    - moving the ball simulates mouse movement
    - squashing the ball causes a left click
    - squishing the ball causes a right click

Ultrasound settings:
    - Example /w Philips SDR-1200:
        - Amplification of entire field: 60 (out of 60)
        - Amplification of near field: 10
        - Amplification of the far field: 0.0
        - Focus: Middle, Far (F1)

NOTE:
    - might require some fiddling with the values
    - is not the most robust, especially if the sizing, and ultrasound machine settings are different

"""

MIN_AREA_SIZE = 5000
MIN_CLICK_REQUIREMENT = 20
MIN_DEFAULT_ELLIPSE_REQUIREMENT = 20
PROBE_ARTIFACT = 30

SQUISH_DISTORTION_THRESHOLD = 0.3
SQUASH_DISTORTION_THRESHOLD = 0.1
SQUASH_SCALE_THRESHOLD = 0.15

# TODO: ignore too small changes in the contour/ellipse

class Deformer(Demo):
    def __init__(self) -> None:
        super().__init__()
        self.monitor_w = 0
        self.monitor_h = 0
        self.set_monitor_dimensions() # evtl. use package screeninfo

        self.default_majors = []
        self.default_minors = []
        self.default_major = None
        self.default_minor = None
        self.default_angle = None

        self.has_squash_clicked = False
        self.has_squish_clicked = False
        self.squish = 0
        self.squash = 0

        self.slider_max = 20 # 255
        self.slider_value = 1 # 50

        self.us_area_threshold = 30

    # evtl. use package screeninfo
    def set_monitor_dimensions(self) -> tuple[int]:
        self.monitor_w, self.monitor_h = pyautogui.size()
    
    def do(self, frame:MatLike, masked:MatLike) -> MatLike:
        """Fit a ellipse over the contours to approximate the shape and compare axis"""
        super().do(frame, masked)

        # masked = cv2.rotate(masked, cv2.ROTATE_90_CLOCKWISE) # TODO rotate image for sideways usage

        # mask top
        cv2.rectangle(masked, (self.us_area.x, self.us_area.y), (self.us_area.x+self.us_area.w, self.us_area.y+PROBE_ARTIFACT), COLORS.BLACK, -1)

        # NOTE: Alternatively: Threshold
        # thresh, ret = cv2.threshold
        # gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (5,5), 0) 
        # _, thresh = cv2.threshold(gray, self.slider_value, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(frame, contours, -1, COLORS.PURPLE, 1)
        # return thresh
        
        # -- SEGMENTATION -- #

        contours, frame = self.segment(masked)
        if len(contours) < 1: return frame

        # get the biggest contour(s) 
        biggest = contours[-1]

        # ignore too small contours
        if cv2.contourArea(biggest) < MIN_AREA_SIZE: return frame

        # concatenate 2nd biggest contour for better contours
        if len(contours) >= 2 and cv2.contourArea(contours[-2]) > 500:
            biggest = np.concatenate((biggest, contours[-2]))

        # draw biggest contour(s)
        if self.is_debug:
                cv2.drawContours(frame, biggest, -1, COLORS.RED, 2)

        # -- DETECTION -- #

        # fit a ellipse around the contour to approximate the shape of the ball   
        ellipse = cv2.fitEllipse(biggest)
        (center_x, center_y), (axis1, axis2), angle = ellipse
        cv2.ellipse(frame, ellipse,COLORS.BLUE,2)
        cv2.circle(frame, (int(center_x), int(center_y)), 5, COLORS.BLUE, -1) 

        major = max(axis1, axis2)
        minor = min(axis1, axis2)

        # ===== INTERACTION ===== #

        # -- MOUSE Interaction -- # 
        
        if not self.is_debug:
            mouse_x, mouse_y = self.translate_mouse_coordinates(center_x, center_y)
            mouse.move(mouse_x, mouse_y, absolute=True, duration=0)

        # -- CLICK Interaction -- #

        # Get default values
        has_set = self.set_default_ellipse(major, minor, angle)
        if not has_set: return frame

        # draw default ellipse
        default_ellipse = (center_x, center_y), (self.default_major, self.default_minor), self.default_angle
        cv2.ellipse(frame, default_ellipse, COLORS.BLACK, 2)
        default_text = f"set default ellipse (black) - reset with 'r'"
        self.write_text(frame, default_text, (10, 40))
    
        distortion = np.log(major / minor)
        scale_log = np.log((major + minor) / (self.default_major + self.default_minor))
        scale_log = np.abs(scale_log)

        # NOTE: Values can be very situation -> might require adapting
        # if aspect_ratio > 1.2 and not self.has_squish_clicked:
        if distortion > SQUISH_DISTORTION_THRESHOLD and not self.has_squish_clicked: 
            # SQUISH
            cv2.ellipse(frame, ellipse,COLORS.GREEN, 2) # feedback
            if self.squish > MIN_CLICK_REQUIREMENT: # HACK: would be better to make framerate dependent
                self.has_squish_clicked = True
            self.squish += 1
        # elif aspect_ratio < 1.1 and scale > 1.2 and not self.has_squash_clicked:
        elif distortion < SQUASH_DISTORTION_THRESHOLD and scale_log > SQUASH_SCALE_THRESHOLD and not self.has_squash_clicked:
            # SQUASH
            cv2.ellipse(frame, ellipse,COLORS.PURPLE, 2) # feedback
            if self.squash > MIN_CLICK_REQUIREMENT:
                self.has_squash_clicked = True
            self.squash += 1
        else:
            if self.has_squish_clicked:
                self.has_squish_clicked = False
                # print("RIGHT CLICK")
                mouse.right_click()
            elif self.has_squash_clicked:
                self.has_squash_clicked = False
                # print("LEFT CLICK")
                mouse.click()
                
            # reset if back to "default" state
            self.squish = 0
            self.squash = 0

        return frame
    
    def set_default_ellipse(self, major, minor, angle):
        """Gather the major and minor axis for a set amount of time and then set their average as default"""
        if len(self.default_majors) < MIN_DEFAULT_ELLIPSE_REQUIREMENT:
            self.default_minors.append(minor)
            self.default_majors.append(major)
            return False
        elif self.default_major == None:
            self.default_major = int(np.average(self.default_majors))
            self.default_minor = int(np.average(self.default_minors))
            self.default_angle = angle
        return True
    
    # use actual us area to scale mouse coordinates
    def translate_mouse_coordinates(self, x:int, y:int)-> tuple[int]:
        """Translate the mouse coordinate to fit the current screen,
        while taking into account, that the US area does not start at (0,0)"""

        if self.us_area == None: return
        if self.monitor_w == None: self.set_monitor_dimensions()

        mouse_x = (x - self.us_area.x) * (self.monitor_w / self.us_area.w)
        mouse_y = (y - self.us_area.y) * (self.monitor_h / self.us_area.h)
        return mouse_x, mouse_y
    
    # ----- other DEMO Class functions ----- #
    
    def reset(self):
        """Reset the default ellipse parameters"""
        self.default_majors = []
        self.default_minors = []
        self.default_major = None
        self.default_minor = None
        self.default_angle = None
        print("resetting the default ellipse")

# ----- MAIN ----- #
video = "../Data/stress_x1.mp4"
# video = "../Data/stress_x15.mp4" # alt video
player = Player(Deformer(), video)
player.start_player()

# ---------------------------------