import cv2
from cv2.typing import MatLike
from Helpers.Demo_Class import Demo
from Helpers.Player import Player

"""
MAIN Implementation of Painter demo
"""

class Painter(Demo):
    def __init__(self) -> None:
        print("[DEMO] -", self.get_name())
        pass

    def do(self, frame:MatLike, gray:MatLike)-> MatLike:
        return frame
        # TODO...


# ----- MAIN ----- #

player = Player(Painter())
player.start_player()
