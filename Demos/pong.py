import cv2
import math
import random
import numpy as np
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

DEGREES_90 = 90

class Game(Demo):
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
        self.score = 0

        self.x0 = 250 # starting point
        self.y0 = 0 # starting point
        self.x = self.x0
        self.y = self.y0
        self.reflection_angle = math.radians(DEGREES_90)
        self.collision_line = []
        self.reflection_line = []
        # self.direction = DOWN

    # ----- GAME LOGIC ----- #

    def update(self, area_height, area_width, contours):
        if self.y > SENSOR_ARTIFACT_AREA: # ignore sensor-artifacts at the top
            self.check_collision(contours)

        self.x, self.y = self.calculate_new_xy((self.x, self.y), self.speed, self.reflection_angle)
        self.check_oob(area_height, area_width)

    def check_oob(self, area_height, area_width):
        """Check if the Ball is out of bounds"""
        # Bottom
        if self.y >= area_height:
            self.reset()
        
        # top = GOAL
        if self.y <= 0:
            self.score += 1
            self.reset()

        # sides
        if self.x > area_width or self.x <= 0:
            # also do reflection ??
            self.reset() 
            print("OOB @ sides")

    def check_collision(self, contours):
        """Check if the ball collides with a found contour"""

        if len(contours) < 1: return

        for contour in contours:
            result = cv2.pointPolygonTest(contour, (self.x, self.y), False) 
            # if result == 0: # direct auf Linie -> nicht reliable
            if result >= 1: 
                # find the closest point on the contour
                contour = contour.squeeze()
                point = (self.x, self.y)
                closest = self.closest_point(point, contour)

                # use an offset to get a line of the contour on which the point lies
                start = contour[closest-10] # TODO handle out of range, offset
                end = contour[closest+10]
                self.collision_line[:] = [start, end] # for drawing lines

                # calculate the slope of the collision line
                collision_slope = self.get_slope(start[0], start[1], end[0], end[1])

                # get incident line
                incident_slope = self.get_slope(self.x0, self.y0, self.x, self.y)

                # get angle between incident line and collision line
                between_angle = self.get_angle_between_lines(collision_slope, incident_slope) # in degrees

                # get incident angle = angle between normal and incident line
                incident_angle = DEGREES_90 - between_angle

                # ?? get reflection angle
                self.reflection_angle = np.deg2rad(incident_angle) * -1

                ref_x, ref_y = self.calculate_new_xy((self.x, self.y), self.speed, self.reflection_angle)
                self.reflection_line[:] = [[self.x, self.y], [ref_x, ref_y]]
    
    # ----- MATH HELPERS ----- #

    def closest_point(self, point: tuple, contour:np.array) -> int:
        """find the closest point in the contour, via: https://codereview.stackexchange.com/q/28207"""
        contour = np.asarray(contour)
        dist_2 = np.sum((contour - point)**2, axis=1)
        return np.argmin(dist_2)
    
    def calculate_new_xy(self, old_xy:tuple, speed, angle_in_radians):
        old_x, old_y = old_xy
        new_x = int(old_x + (speed*math.cos(angle_in_radians)))
        new_y = int(old_y + (speed*math.sin(angle_in_radians)))
        # print("new xy", new_x, new_y)
        return new_x, new_y
    
    def get_slope(self, start_x, start_y, end_x, end_y):
        """Return the slope of a line"""
        if (end_x - start_x) == 0: return 0 # vertical line
        return (end_y - start_y) / (end_x - start_x)
    
    def get_angle_between_lines(self, slope1, slope2):
        """Get the angle between two lines in degrees, via: https://stackoverflow.com/a/57503229"""
        return math.degrees(math.atan((slope2-slope1)/(1+(slope2*slope1))))

    # ----- GAME HELPERS ----- # 

    def reset(self):
        self.x = self.x0
        self.y = self.y0
        self.reflection_angle = math.radians(DEGREES_90)
        # self.direction = DOWN#
        # self.x = random.randint(80, 400) # TODO offset to keep in play area
        # self.radius = random.randint(5, 20)
        # self.speed = random.randint(5, 15)

    def draw(self, frame, debug=False)-> None:
        cv2.circle(frame, (self.x, self.y), self.radius, self.color, -1)        
        self.write_score(frame)

        # DEBUG
        if len(self.collision_line) != 0:
            start = (self.collision_line[0][0], self.collision_line[0][1])
            end = (self.collision_line[1][0], self.collision_line[1][1])
            cv2.line(frame, start, end, COLORS.RED, 3)

        if len(self.reflection_line) != 0:
            cv2.arrowedLine( frame, (self.reflection_line[0][0], self.reflection_line[0][1]), 
                            (self.reflection_line[1][0] + 10, self.reflection_line[1][1] + 10),
                            color=(0, 255, 0), thickness=2, tipLength=0.3)

    def write_score(self, frame):
        text = f"Score: {self.score}"
        pos = (40, 50)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_PLAIN, 1, COLORS.GREEN, 1, cv2.LINE_AA)    

# ----- MAIN ----- #

video = "../Data/stressball.mp4"
player = Player(Game(), video)
player.start_player()
