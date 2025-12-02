import cv2
from cv2.typing import MatLike
from Helpers.Demo_Class import Demo
from Helpers.Player import Player
from Helpers.Parameters import COLORS

"""
MAIN implementation of Painter demo
"""

BBOX_MIN_W = 10 # so that it doesn't detect bubbles
BBOX_MIN_H = 10

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

    def do(self, frame:MatLike, masked:MatLike)-> MatLike:
        super().do(frame, masked)

        # NOTE threshold might work better for some videos
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.slider_value, THRESH_MAX, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # contours, frame = self.segment(masked)

        if len(contours) == 0:
            self.is_newline = True
        else:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.drawContours(frame, contours, -1, COLORS.PURPLE, 1)

            if w > BBOX_MIN_W and h > BBOX_MIN_H:
                # find the center
                cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS.GREEN, 1)
                center = (round(x + w/2), round(y + h/2))
                cv2.circle(frame, center, 5, COLORS.GREEN, -1)

                if self.is_newline:
                    self.current_line = Line()
                    self.lines.append(self.current_line)
                    self.is_newline = False

                self.current_line.add_point(center)
        
        self.draw_lines(frame)
        return frame

    def draw_lines(self, frame:MatLike):
        if len(self.lines) == 0: return

        for line in self.lines:
            line.draw(frame)

class Line():
    """Handles operations on all points of a single line"""

    def __init__(self) -> None:
        self.points = []

    def add_point(self, point:tuple[int]):
        self.points.append(point)

    def draw(self, frame):
        if len(self.points) == 0:
            return

        for i in range (0, len(self.points)-1):
            start = self.points[i]
            end = self.points[i+1]
            cv2.line(frame, start, end, COLORS.RED, 3)
 

# ----- MAIN ----- #
video = "../Data/heart.mp4"
# video = "../Data/zigzag.mp4"
# video = "../Data/draw-triangle.mp4"
player = Player(Painter(), video)
player.start_player()
