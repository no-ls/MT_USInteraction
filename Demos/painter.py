import cv2
from cv2.typing import MatLike
import numpy as np
from Helpers.Demo_Class import Demo
from Helpers.Player import Player
from Helpers.Parameters import COLORS
import random

"""
MAIN implementation of Painter demo
"""

# TODO make adjustable ?
BBOX_MIN_W = 10 # so that it doesn't detect bubbles
BBOX_MIN_H = 10

MAX_AREA_SIZE = 2000
MAX_LINE_STRENGTH = 10

DEFAULT_LINE_WIDTH = 3
DEFAULT_THRESH = 130
THRESH_MAX = 255

# NOTE: some videos work better with threshold, some with color quantization

class Painter(Demo):
    def __init__(self) -> None:
        super().__init__()
        self.slider_value = DEFAULT_THRESH
        self.slider_max = THRESH_MAX
        self.slider_name = "Threshold"
        self.is_newline = True
        self.lines: list[Line] = []
        self.current_line: Line = None
        self.line_color = COLORS.RED

    def do(self, frame:MatLike, masked:MatLike)-> MatLike:
        super().do(frame, masked)
        frame = masked

        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.slider_value, THRESH_MAX, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # contours, frame = self.segment(masked)

        if len(contours) == 0: # doesn't always work for new_lines
            self.is_newline = True
        else:
            max_c = max(contours, key = cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_c)

            if w > BBOX_MIN_W and h > BBOX_MIN_H:
                line_strength = self.map_line_strength(max_c)
                center = (round(x + w/2), round(y + h/2))

                if self.is_debug:
                    text = f"{x}, {y}"
                    self.write_text(frame, text, (x, y))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS.PURPLE, 2)
                    cv2.circle(frame, center, 5, COLORS.GREEN, -1)
                    cv2.drawContours(frame, contours, -1, COLORS.BLUE, 1)

                if self.is_newline:
                    self.current_line = Line(self.line_color)
                    self.lines.append(self.current_line)
                    self.is_newline = False

                self.current_line.add_point(center, line_strength)
            else:
                self.line_color = self.get_random_color()
                # NOTE: immediate new_lining causes small breaks in line
                self.is_newline = True
        
        self.draw_lines(frame)
        return frame
    
    def map_line_strength(self, contour):
        """Uses the contour area to find a value for the line thickness"""
        area = int(cv2.contourArea(contour))
        if area > MAX_AREA_SIZE:
            return MAX_LINE_STRENGTH
        return int(np.interp(area, [0, MAX_AREA_SIZE], [1, MAX_LINE_STRENGTH]))
    
    def get_random_color(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (b, g, r)

    def draw_lines(self, frame:MatLike):
        if len(self.lines) == 0: return

        for line in self.lines:
            line.draw(frame)

    def reset(self):
        """Reset all existing lines and start new"""
        self.lines = []
        self.is_newline = True

# TODO: save option (w/w.o US background) ?


class Line():
    """Handles operations on all points of a single line"""

    def __init__(self, color:tuple[int]=COLORS.RED) -> None:
        self.points = []
        self.line_strengths = []
        self.color = color

    def add_point(self, point:tuple[int], line_strength:int=DEFAULT_LINE_WIDTH):
        self.points.append(point)
        self.line_strengths.append(line_strength)

    def draw(self, frame):
        if len(self.points) == 0:
            return

        for i in range (0, len(self.points)-1):
            start = self.points[i]
            end = self.points[i+1]
            cv2.line(frame, start, end, self.color, self.line_strengths[i])
 

# ----- MAIN ----- #
# video = "../Data/heart.mp4"
# video = "../Data/zigzag.mp4"
video = "../Data/draw_new.mp4"
player = Player(Painter(), video)
player.start_player()
