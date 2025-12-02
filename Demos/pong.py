import cv2
import random
from cv2.typing import MatLike
from Helpers.Demo_Class import Demo, grayscale
from Helpers.Player import Player
from Helpers.Parameters import COLORS

"""
A simple Pong-type game
"""

SENSOR_ARTIFACT_AREA = 100
DOWN = 1
UP = -1

class Visualizer(Demo):
    def __init__(self) -> None:
        super().__init__()
        self.pong = Pong()

    def do(self, frame: MatLike, masked: MatLike) -> MatLike:
        super().do(frame, masked)

        contours, frame = self.segment(masked)
        self.do_pong(frame, contours)

        return frame
    
    def do_pong(self, frame:MatLike, contours):
        self.pong.update(self.area_h, self.masked_w, contours)
        self.pong.draw(frame)

# ----- PONG ----- #

UP = -1
DOWN = 1

class Pong():
    def __init__(self) -> None:
        self.speed = 8
        self.radius = 10
        self.color = COLORS.GREEN
        self.x = 250 # TODO
        self.y = 0
        self.direction = DOWN

    def update(self, area_height, area_width, contours):
        if self.y > SENSOR_ARTIFACT_AREA: # ignore sensor-artifacts at the top
            self.check_collision(contours)
        self.y += self.speed * self.direction

        # TODO: check brighness-values behind ball -> if hypoechoic make (slightly) smaller (once)

        # CHECK out-of-bounds
        if self.y >= area_height or self.y <= 0: # reset when oob
            if self.y <= 0:
                print("+1")
            self.reset()

    def check_collision(self, contours):
        if len(contours) < 1: return

        for contour in contours:
            result = cv2.pointPolygonTest(contour, (self.x, self.y), False) 
            if result == 1:
                # ??: calculate angle of reflection 
                    # -> Winkel von Ball & Winkel von Kontur = Winkel abprallen
                self.direction = UP

    def reset(self):
        self.y = 0
        self.direction = DOWN
        self.x = random.randint(80, 400) # TODO offset to keep in play area
        self.radius = random.randint(5, 20)
        self.speed = random.randint(5, 15)

    def draw(self, frame, debug=False)-> None:
        cv2.circle(frame, (self.x, self.y), self.radius, self.color, -1)        

# ----- MAIN ----- #

video = "../Data/stressball.mp4"
player = Player(Visualizer(), video)
player.start_player()
