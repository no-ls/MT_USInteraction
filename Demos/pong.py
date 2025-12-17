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
DEGREES_180 = 180

OFFSET_COLLISION_END_POINTS = 10 # How much to go to each side of the collision point on the contour

class Game(Demo):
    def __init__(self) -> None:
        super().__init__()
        self.us_area_threshold = 20
        self.pong = Pong()

    def do(self, frame: MatLike, masked: MatLike) -> MatLike:
        super().do(frame, masked)

        contours, frame = self.segment(masked)
        self.do_pong(frame, contours)

        return frame
    
    def do_pong(self, frame:MatLike, contours):
        if self.image_h == None: pass
        self.pong.update(self.image_h, self.image_w, contours)
        self.pong.draw(frame, self.is_debug)

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
        self.incidence_start = [self.x0, self.y0]
        self.reflection_angle = math.radians(DEGREES_90)
        self.collision_line = []
        self.reflection_line = []
        self.incidence_line = []
        self.has_reflected = False # so it doesn't bounce inside the contour
        # self.direction = DOWN

    # ----- GAME LOGIC ----- #

    def update(self, area_height, area_width, contours):
        if self.y > SENSOR_ARTIFACT_AREA: # ignore sensor-artifacts at the top
            self.check_collision(contours)

        self.x, self.y = self.calculate_new_xy((self.x, self.y), self.speed, self.reflection_angle)
        self.check_oob(area_height, area_width)

    def check_oob(self, area_height, area_width):
        """Check if the Ball is out of bounds of the screen"""

        # Bottom
        if self.y >= area_height:
            self.reset()
        else: # also reflect
            start = [0, 0]
            end = [area_width, area_height]

            if self.y <= 0: # top = GOAL
                self.score += 1
                end[:] = [area_width, 0]
                self.reset(soft_reset=False)
                # self.reset(soft_reset=True)
            # NOTE: does not always work -> but usually gets out of bounds anyway
            elif self.x > area_width: # right side
                start[:] = [area_width, 0]
                end[:] = [area_width, area_height]
                self.calculate_reflection(start, end)
            elif self.x <= 0: # left side
                end[:] = [0, area_height]
                self.calculate_reflection(start, end)

    def check_collision(self, contours):
        """Check if the ball collides with a found contour"""

        if len(contours) < 1: return

        for contour in contours:
            result = cv2.pointPolygonTest(contour, (self.x, self.y), False) 

            # TODO implement failsave -> too many reflections aka, if got stuck

            if result >= 1 and not self.has_reflected: # i.e on the line or inside the contour
                self.has_reflected = True

                # find the point on the contour that is closest to the collision point
                contour = contour.squeeze()
                point = (self.x, self.y)
                closest = self.closest_point(point, contour)

                # handle out of range indices
                start_idx = closest - OFFSET_COLLISION_END_POINTS
                if start_idx < 0:
                    start_idx = 0

                end_idx = closest + OFFSET_COLLISION_END_POINTS
                if end_idx >= len(contour):
                    end_idx = len(contour) - 1

                # use an offset to get a line of the contour on which the point lies
                start = contour[start_idx] 
                end = contour[end_idx]

                self.calculate_reflection(start, end)
            elif result == -1 and self.has_reflected:
                self.has_reflected = False
                print("reset reflection")

    def calculate_reflection_1(self, start:list[int], end:list[int], direction=-1):
        """TEST
        via: ChatGPT (I have the start and endpoints for two lines, the incidence line and the mirror line. 
        How do I calculate the angle of reflection, so that it works from any direction. I'm using python)"""
        # incidence = self.incidence_start[0], self.incidence_start[1], self.x, self.y
        # collision start[0], start[1], end[0], end[1]
        P1 = np.array([self.incidence_start[0], self.incidence_start[1]])
        P2 = np.array([self.x, self.y])
        M1 = np.array([start[0], start[1]])
        M2 = np.array([end[0], end[1]])

        incident = P2 - P1
        mirror = M2 - M1

        # normalize vectors
        incident = incident / np.linalg.norm(incident)
        mirror = mirror / np.linalg.norm(incident)

        # compute normal of mirror
        normal = np.array([-mirror[1], mirror[0]])
        normal = normal / np.linalg.norm(normal)

        print("normal", normal)
        self.speed = - 8

        # reflect
        dot = np.dot(incident, normal)
        reflected = incident - 2 * dot * normal

        # angles 
        angle_incidence = np.arccos(np.clip(dot, -1, 1))
        angle_reflection = angle_incidence 

        theta_ref = np.arctan2(reflected[1], reflected[0])
        # set
        self.reflection_angle = np.deg2rad(theta_ref)

        self.collision_line[:] = [start, end]

    def calculate_reflection(self, start:list[int], end:list[int], direction=-1):
        """Calculate the reflection off of a line"""

        # calculate the slope of the collision line
        collision_slope = self.get_slope(start[0], start[1], end[0], end[1])

        # get incident line
        incident_slope = self.get_slope(self.incidence_start[0], self.incidence_start[1], self.x, self.y)

        # get angle between incident line and collision line
        between_angle = self.get_angle_between_lines(collision_slope, incident_slope) # in degrees

        # get incident angle = angle between normal and incident line
        incidence_angle = DEGREES_90 - between_angle

        # get reflection angle
        mirror_angle = DEGREES_180 - incidence_angle # mirror the incidence angle
        self.reflection_angle = np.deg2rad(mirror_angle) * -1 # make it go up
        
        # for DEBUG drawing
        self.collision_line[:] = [start, end]

        ref_x, ref_y = self.calculate_new_xy((self.x, self.y), self.speed, self.reflection_angle)
        self.reflection_line[:] = [[self.x, self.y], [ref_x, ref_y]]

        self.incidence_line[:] = [self.incidence_start, [self.x, self.y]]

        # reset incidence_line start
        self.incidence_start = [self.x, self.y]
            
    # ----- MATH HELPERS ----- #

    def closest_point(self, point: tuple, contour:np.array) -> int:
        """find the closest point in the contour, via: https://codereview.stackexchange.com/q/28207"""
        contour = np.asarray(contour)
        dist_2 = np.sum((contour - point)**2, axis=1)
        return np.argmin(dist_2)
    
    def calculate_new_xy(self, old_xy:tuple, speed, angle_in_radians):
        """Use an angle to calculate the new coordinates, via: https://stackoverflow.com/a/46697552"""
        old_x, old_y = old_xy
        new_x = int(old_x + (speed*math.cos(angle_in_radians)))
        new_y = int(old_y + (speed*math.sin(angle_in_radians)))
        return new_x, new_y
    
    def get_slope(self, start_x, start_y, end_x, end_y):
        """Return the slope of a line"""
        if (end_x - start_x) == 0: return 0 # vertical line
        return (end_y - start_y) / (end_x - start_x)
    
    def get_angle_between_lines(self, slope1, slope2):
        """Get the angle between two lines in degrees, via: https://stackoverflow.com/a/57503229"""
        return math.degrees(math.atan((slope2-slope1)/(1+(slope2*slope1))))

    # ----- GAME HELPERS ----- # 

    def reset(self, soft_reset=False):
        if not soft_reset:
            self.x = self.x0
            self.y = self.y0
        self.reflection_angle = math.radians(DEGREES_90)
        self.speed = 8
        self.incidence_start = [self.x0, self.y0]
        # self.direction = DOWN#
        # self.x = random.randint(80, 400) # TODO offset to keep in play area
        # self.radius = random.randint(5, 20)
        # self.speed = random.randint(5, 15)

    def draw(self, frame, debug=False)-> None:
        cv2.circle(frame, (self.x, self.y), self.radius, self.color, -1)        
        self.write_score(frame)

        # DEBUG
        if debug:
            self.draw_debug_lines(frame)
                
    def draw_debug_lines(self, frame):
        """draw the angles and reflection lines to help understand whats happening"""
        if len(self.collision_line) != 0:
            start = (self.collision_line[0][0], self.collision_line[0][1])
            end = (self.collision_line[1][0], self.collision_line[1][1])
            cv2.line(frame, start, end, COLORS.RED, 3)
       
        if len(self.incidence_line) != 0:
            cv2.arrowedLine( frame, (self.incidence_line[0][0], self.incidence_line[0][1]), 
                            (self.incidence_line[1][0], self.incidence_line[1][1]),
                            color=COLORS.PURPLE, thickness=2, tipLength=0.3)

        if len(self.reflection_line) != 0:
            cv2.arrowedLine( frame, (self.reflection_line[0][0], self.reflection_line[0][1]), 
                            (self.reflection_line[1][0], self.reflection_line[1][1]),
                            color=(0, 255, 0), thickness=1, tipLength=0.3)
            

    def write_score(self, frame):
        text = f"Score: {self.score}"
        pos = (40, 50)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_PLAIN, 1, COLORS.GREEN, 1, cv2.LINE_AA)    

# ----- MAIN ----- #

video = "../Data/stressball.mp4"
player = Player(Game(), video)
player.start_player()
