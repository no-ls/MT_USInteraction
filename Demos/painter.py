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

class Painter(Demo):
    def __init__(self) -> None:
        print("[DEMO] -", self.get_name())
        self.lines = []

    def do(self, frame:MatLike, gray:MatLike)-> MatLike: 
        self.show_fps(frame)

        # blur image?
        # gray = cv2.bilateralFilter(gray,9,75,75)
        # gray = cv2.medianBlur(gray, 5)

        # NOTE threshold might change for videos
        _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        # draw bounding box
        if w > BBOX_MIN_W and h > BBOX_MIN_H:
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS.GREEN, 1)

            # draw center 
            center = (round(x + w/2), round(y + h/2))
            cv2.circle(frame, center, 5, COLORS.GREEN, -1)
            self.lines.append(center)

        self.draw_lines(frame)
        return frame

    def draw_lines(self, frame:MatLike):
        if len(self.lines) == 0:
            return
        
        # TODO: restart line, if finger removed
        
        for i in range (0, len(self.lines)-1):
            start = self.lines[i]
            end = self.lines[i+1]
            cv2.line(frame, start, end, COLORS.RED, 3)


# ----- MAIN ----- #
video = "../Data/heart.mp4"
# video = "../Data/zigzag.mp4"
# video = "../Data/smiley.mp4"
player = Player(Painter(), video)
player.start_player()
