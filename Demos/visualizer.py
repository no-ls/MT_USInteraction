import cv2
from cv2.typing import MatLike
from Helpers.Demo_Class import Demo, grayscale
from Helpers.Player import Player
from Helpers.Parameters import COLORS

"""
MAIN Implementation of the visualizer demo
- [ ] simulate sound waves traveling ))) and reflecting (((
    - [x] prototype with single circle
- [x] ! simulate strength of reflection and propagation !
    - brighter == stronger reflection
- [ ] ...
"""

class REFLECTION_STRENGTH():
    NULL = 0
    HYPER = 0.2
    HYPO = 0.1
    NOISE = 0.05

DOWN = 1
UP = -1

class Visualizer(Demo):
    def __init__(self) -> None:
        super().__init__()
        self.waves: list[Sound_Wave] = []
        self.init_simulation()

    def do(self, frame: MatLike, masked: MatLike) -> MatLike:
        super().do(frame, masked)

        # contours, frame = self.segment(masked)

        self.simulate_sound_wave(frame)
                
        return frame
    
    def simulate_sound_wave(self, frame:MatLike)-> None:
        for wave in self.waves:
            wave.update_position(self.area_h)
            wave.check_brightness(frame)
            wave.simulate_reflection()
            wave.draw_reflections(frame, self.area_h)
            wave.draw(frame, self.is_debug)

    def init_simulation(self)-> None:
        wave = Sound_Wave(250, 0, 10)
        self.waves.append(wave)

# ----- SOUND WAVE ----- #

class Sound_Wave():
    def __init__(self, x, y, radius, direction=DOWN, color=COLORS.BLACK) -> None:
        self.speed = 5
        # self.area_w = None
        # self.area_h = None
        self.text = "Null"
        self.reflection = REFLECTION_STRENGTH.NULL
        self.prev_reflection = REFLECTION_STRENGTH.NULL

        self.x = x
        self.y = y
        self.radius = radius
        self.direction = direction
        self.color = color

        self.reflections: list[Sound_Wave] = []

    def draw(self, frame, debug=False)-> None:
        cv2.circle(frame, (self.x, self.y), self.radius, self.color, -1)
        if debug:
            cv2.putText(frame, self.text, (self.x+5, self.y-5), cv2.FONT_HERSHEY_PLAIN, 1, COLORS.WHITE, 1, cv2.LINE_AA)

    # TODO: better out of bounds check -> only apply to masked area (aka ignore black)
    def update_position(self, area_height)-> None:
        """Update the position of a soundwave and make out-of-bounds checks"""
        self.y += self.speed * self.direction
        if self.y >= area_height: # reset for oob
            self.y = 0

    def check_brightness(self, frame:MatLike):
        """Evaluate the pixels based on their brightness value"""

        frame = grayscale(frame)
        v = frame[self.y][self.x]
        # print("px = ", frame[self.y][self.x])

        # NOTE: Currently very simplified
        if v == 0:
            self.set_reflection_attributes(REFLECTION_STRENGTH.NULL, COLORS.BLACK, "Null")
        if v > 0 and v < 30:
            self.set_reflection_attributes(REFLECTION_STRENGTH.NOISE, COLORS.BLUE, "Noise")
        if v >= 30 and v < 80:
            self.set_reflection_attributes(REFLECTION_STRENGTH.HYPO, COLORS.GREEN, "Hypo")
        if v >= 80:
            self.set_reflection_attributes(REFLECTION_STRENGTH.HYPER, COLORS.YELLOW, "Hyper")
        # TODO: fix magic numbers

    def set_reflection_attributes(self, strength, color, text):
        self.color = color
        self.reflection = strength
        self.text = text

    # ----- Handle Reflections ----- #

    def simulate_reflection(self):
        """Simulate the (partial- ) reflection at a boundary based on the given strength"""
        if self.reflection == REFLECTION_STRENGTH.NULL: return
        if self.reflection == self.prev_reflection: return

        # calculate portion of sound wave to reflect and subtract it from the original
        part = int(self.radius * self.reflection)
        self.radius -= part
        # print(f"{part} of {self.radius}")

        if part == 0: part = 1 # for visibility
        reflection = Sound_Wave(self.x, self.y, part, UP, self.color)
        self.reflections.append(reflection)
        
        self.prev_reflection = self.reflection

    def draw_reflections(self, frame, area_h):
        for reflection in self.reflections:
            reflection.update_position(area_h)
            reflection.draw(frame)
        

# ----- MAIN ----- #

video = "../Data/stressball.mp4"
player = Player(Visualizer(), video)
player.start_player()
