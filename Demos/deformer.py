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

# How long/short the axises of the ellipse are to differentiate
SQUISH_AXIS_LENGTH = 80
SQUASH_AXIS_LENGTH = 300 

MIN_AREA_SIZE = 5000

AXIS_OFFSET = 2

MIN_CLICK_REQUIREMENT = 50

# TODO: ignore too small changes in the contour/ellipse

class Deformer(Demo):
    def __init__(self) -> None:
        super().__init__()
        self.base_distance = 0
        self.i = 0
        self.slider_value = 1
        self.has_clicked = False
        self.has_squash_clicked = False
        self.has_squish_clicked = False

        self.prev_minor = 0
        self.prev_major = 0

        self.monitor_w = 0
        self.monitor_h = 0
        self.set_monitor_dimensions() # evtl. use package screeninfo

        self.prev_major = None
        self.prev_minor = None
        self.prev_diff = None
        self.is_spasm = False

        self.default_axises = []
        self.default_axis = None

        self.default_majors = []
        self.default_minors = []
        self.default_major = None
        self.default_minor = None
        self.default_angle = None

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
        # TODO: make reset-able (r)
        if len(self.default_majors) < 10:
            self.default_minors.append(minor)
            self.default_majors.append(major)
            return frame
        elif self.default_major == None:
            self.default_major = int(np.average(self.default_majors))
            self.default_minor = int(np.average(self.default_minors))
            self.default_angle = angle

        # draw default ellipse
        default_ellipse = (center_x, center_y), (self.default_major, self.default_minor), self.default_angle
        cv2.ellipse(frame, default_ellipse, COLORS.BLACK, 2)
        default_text = f"set default ellipse (black) - reset with 'r'"
        self.write_text(frame, default_text, (10, 40))
        
        # get normalized values for better (size independent) comparison
        aspect_ratio = major / minor
        scale = (major + minor) / (self.default_major + self.default_minor)
        self.write_text(frame, f"R: {round(aspect_ratio, 2)} / S: {round(scale, 2)}", (int(center_x), int(center_y)))

        if aspect_ratio > 1.2 and not self.has_squish_clicked: 
            # SQUISH
            cv2.ellipse(frame, ellipse,COLORS.GREEN, 2) # feedback
            if self.squish > MIN_CLICK_REQUIREMENT: # TODO make framerate dependent
                self.has_squish_clicked = True
            self.squish += 1
        elif aspect_ratio < 1.1 and scale > 1.2 and not self.has_squash_clicked:
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

        # print(self.squish, self.squash)
        return frame
    
    def translate_mouse_coordinates_old(self, x:int, y:int)-> tuple[int]:
        """Translate the coordinates to fit the current screen"""
        if self.image_h == None or self.monitor_w == None:
            self.set_monitor_dimensions()

        # scale mouse positions to fit monitor
        mouse_x = x * (self.monitor_w / self.image_w)
        mouse_y = y * (self.monitor_h / self.image_h)
        return mouse_x, mouse_y
    
    # use actual us area to scale mouse coordinates
    def translate_mouse_coordinates(self, x:int, y:int)-> tuple[int]:
        """Translate the mouse coordinate to fit the current screen,
        while taking into account, that the US area does not start at (0,0)"""

        if self.us_area == None: return
        if self.monitor_w == None: self.set_monitor_dimensions()

        mouse_x = (x - self.us_area.x) * (self.monitor_w / self.us_area.w)
        mouse_y = (y - self.us_area.y) * (self.monitor_h / self.us_area.h)
        return mouse_x, mouse_y

# ----- HELPERS -----  #

def get_midpoint(x1, y1, x2, y2):
    # via: https://stackoverflow.com/a/5047796
    return (int((x1 + x2)/2), int((y1 + y2)/2))

def get_distance(x1, y1, x2, y2):
    c1 = np.array((x1, y1))
    c2 = np.array((x2, y2))
    return np.linalg.norm(c1 - c2)
    
def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    """
    via: https://stackoverflow.com/a/69884067
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255*(1-contrast)/2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)

def do_kmeans(value:int, img:MatLike) -> MatLike:
    """via: https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html"""
    
    # downsample frame for faster computation
    resample = 2
    rows, cols, _channels = map(int, img.shape)
    img = cv2.pyrDown(img, dstsize=(cols // resample, rows // resample))

    # blur
    cv2.GaussianBlur(img, (5,5), 0) 

    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    # define criteria, number of clusters(K)
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0)
    K = value
    
    # apply kmeans()
    _, label, center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # upsample for display
    rows, cols, _channels = map(int, res2.shape)
    res2 = cv2.pyrUp(res2, dstsize=(resample * cols, resample * rows))
    
    return res2

def do_color_quantization(img:MatLike, gray:MatLike, value:int=7):
    """via: https://stackoverflow.com/a/66339640 """
    # convert to gray as float in range 0 to 1
    num_colors = value
    gray = gray.astype(np.float32)/255

    # quantize and convert back to range 0 to 255 as 8-bits
    result = 255*np.floor(gray*num_colors+0.5)/num_colors
    result = result.clip(0,255).astype(np.uint8)
    return result

# TODO adjust params to work with different zoom factors
    # e.g. take first occurrence of ball and use as default values

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
# if axis_diff >= (self.default_axis / 2):
#     cv2.ellipse(frame, ellipse,COLORS.GREEN, 2)
#     self.write_text(frame, f"SQUISH: {axis_diff}", (int(center_x), int(center_y)))
#     self.has_squish_clicked = True 
# # SQUASH
# elif major_diff > (self.default_axis / 2) and minor_diff > (self.default_axis / 3):
#     cv2.ellipse(frame, ellipse,COLORS.PURPLE, 2)
#     self.write_text(frame, f"SQUASH: {major_diff}/{minor_diff}", (int(center_x), int(center_y)))
#     self.has_squash_clicked = True
"""

# OLD Version (keep for now...)
def do_old(self, frame: MatLike, masked: MatLike) -> MatLike:

    h, w, _ = masked.shape
    half_h = int(h/2)
    half_w = int(w/2)
    cv2.line(frame, (0, half_h), (h, half_h), COLORS.WHITE, 1, 1)
    cv2.line(frame, (half_w, 0), (half_w, h), COLORS.WHITE, 1, 1)

    masked = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
    q = do_color_quantization(frame, masked, 4)
    _, thresh = cv2.threshold(q, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # PARSE INTERACTION

    # get biggest and second biggest contour, then calculate halfway point
    cnts = sorted(contours, key=cv2.contourArea)
    one = cnts[-1] # biggest contour
    two = cnts[-2] # 2nd biggest
    cv2.drawContours(frame, [one], -1, COLORS.RED, 1)
    cv2.drawContours(frame, [two], -1, COLORS.RED, 1)

    area2 = cv2.contourArea(two)        
    if area2 > 200: 
        m1 = cv2.moments(one)
        m2 = cv2.moments(two)
        if m1["m00"]!= 0 and m2["m00"] != 0:
            c1x = int(m1["m10"] / m1["m00"])
            c1y = int(m1["m01"] / m1["m00"])
            c2x = int(m2["m10"] / m2["m00"])
            c2y = int(m2["m01"] / m2["m00"])

            mid = get_midpoint(c1x, c1y, c2x, c2y) # middle of 1st and 2nd biggest areas
            distance = get_distance(c1x, c1y, c2x, c2y)
            if distance < 100:
                print("squish", distance)
            if distance > 250:
                print("squash", distance)

            cv2.line(frame, (c1x, c1y), (c2x, c2y), COLORS.GREEN, 1, 1) # distance line
            cv2.circle(frame, mid, 5, COLORS.GREEN, -1) # move towards this
            cv2.putText(frame, " position", mid, 1, 1, COLORS.GREEN, 1)

    # Test
    for contour in contours:
        area = cv2.contourArea(contour)
        # only care about the bigger(est) contours
        
        if area > 200: 
            # draw centroid:
            M = cv2.moments(contour)
            if M["m00"]!= 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, COLORS.RED, 2)
                # cv2.drawContours(frame, [contour], -1, COLORS.GREEN, 1)

                # where is centroid
                if cx < half_w and cy < half_h:
                    cv2.putText(frame, " top left", (cx, cy), 1, 1, COLORS.RED, 1)
                elif cx < half_w and cy > half_h:
                    cv2.putText(frame, " bottom left", (cx, cy), 1, 1, COLORS.RED, 1)
                elif cx > half_w and cy < half_h:
                    cv2.putText(frame, " top right", (cx, cy), 1, 1, COLORS.RED, 1)
                else:
                    cv2.putText(frame, " bottom right", (cx, cy), 1, 1, COLORS.RED, 1)

    self.show_fps(frame)
    return frame