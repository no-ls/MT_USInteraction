import cv2
import numpy as np
from Helpers.Demo_Class import Demo
from Helpers.Player import Player
from Helpers.Parameters import COLORS
from cv2.typing import MatLike
import pyautogui
import mouse

"""
NOTEs
- detect ball outline with contours
    - Problem: sometimes the contours are very broken up
    - increasing the contrast with the US machine + using color-quant 1 helps
- fit an ellipse over the contours
    - approximates the shape, but stays flexible
        - other approximations (e.g. circle, square, simplified contour) don't change enough to differentiate
    - SQUISH = compare the two axis of the ellipse, to check if one is sig. longer
    - SQUASH = compare the length of the axis to a (currently) set value; check if its larger
- ? use the center of the ellipse to mimic movement
- find use case:
    - A) as mouse / for mouse-like interaction
    - B) assign controls (e.g. squish = pause) / create simple game
    - C) just visualize the interaction -> e.g. squash increases amount, squish decreases
"""

MIN_AREA_SIZE = 5000
MIN_CLICK_REQUIREMENT = 50
MIN_DEFAULT_ELLIPSE_REQUIREMENT = 20

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

    # evtl. use package screeninfo
    def set_monitor_dimensions(self) -> tuple[int]:
        self.monitor_w, self.monitor_h = pyautogui.size()
    
    def do(self, frame:MatLike, masked:MatLike) -> MatLike:
        """Fit a ellipse over the contours to approximate the shape and compare axis"""
        super().do(frame, masked)
        
        contours, frame = self.segment(masked)
        if len(contours) < 1: return frame

        # get the biggest contour(s) 
        biggest = contours[-1]

        # ignore too small contours
        if cv2.contourArea(biggest) < MIN_AREA_SIZE: return frame

        # concatenate 2nd biggest contour for better contours
        if len(contours) >= 2 and cv2.contourArea(contours[-2]) > 500:
            biggest = np.concatenate((biggest, contours[-2]))

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
        
        # get normalized values for better (size independent) comparison
        # aspect_ratio = major / minor
        # scale = (major + minor) / (self.default_major + self.default_minor)
        # self.write_text(frame, f"R: {round(aspect_ratio, 2)} / S: {round(scale, 2)}", (int(center_x), int(center_y)))

        distortion = np.log(major / minor)
        scale_log = np.log((major + minor) / (self.default_major + self.default_minor))
        self.write_text(frame, f"D: {round(distortion, 2)} / S: {round(scale_log, 2)}", (int(center_x), int(center_y)))

        # if aspect_ratio > 1.2 and not self.has_squish_clicked:
        if distortion > 0.3 and not self.has_squish_clicked: 
            # SQUISH
            cv2.ellipse(frame, ellipse,COLORS.GREEN, 2) # feedback
            if self.squish > MIN_CLICK_REQUIREMENT: # TODO make framerate dependent
                self.has_squish_clicked = True
            self.squish += 1
        # elif aspect_ratio < 1.1 and scale > 1.2 and not self.has_squash_clicked:
        elif distortion < 0.1 and scale_log > 0.35 and not self.has_squash_clicked:
            # SQUASH
            cv2.ellipse(frame, ellipse,COLORS.PURPLE, 2) # feedback
            if self.squash > MIN_CLICK_REQUIREMENT:
                self.has_squash_clicked = True
            self.squash += 1
        else:
            if self.has_squish_clicked:
                self.has_squish_clicked = False
                print("RIGHT CLICK")
                # mouse.right_click()
            elif self.has_squash_clicked:
                self.has_squash_clicked = False
                print("LEFT CLICK")
                # mouse.click()
                
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
# video = "../Data/stressball.mp4"
video = "../Data/stress_deform2.mp4"
# video = "../Data/stress_x1.mp4"
# video = "../Data/stress_x15.mp4"
player = Player(Deformer(), video)
player.start_player()

# ---------------------------------

"""
# SQUISH
if axis_diff >= (self.default_axis / 2):
    cv2.ellipse(frame, ellipse,COLORS.GREEN, 2)
    self.write_text(frame, f"SQUISH: {axis_diff}", (int(center_x), int(center_y)))
    self.has_squish_clicked = True 
# SQUASH
elif major_diff > (self.default_axis / 2) and minor_diff > (self.default_axis / 3):
    cv2.ellipse(frame, ellipse,COLORS.PURPLE, 2)
    self.write_text(frame, f"SQUASH: {major_diff}/{minor_diff}", (int(center_x), int(center_y)))
    self.has_squash_clicked = True

    def translate_mouse_coordinates_old(self, x:int, y:int)-> tuple[int]:
        # Translate the coordinates to fit the current screen
        if self.image_h == None or self.monitor_w == None:
            self.set_monitor_dimensions()

        # scale mouse positions to fit monitor
        mouse_x = x * (self.monitor_w / self.image_w)
        mouse_y = y * (self.monitor_h / self.image_h)
        return mouse_x, mouse_y
"""