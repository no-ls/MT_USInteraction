# MAIN of painter
from Helpers.Demo_Class import Demo
from Helpers.Player import Player


class Painter(Demo):
    def __init__(self) -> None:
        pass

    def do(self):
        print("doing", self.get_name())
        # TODO


# ----- MAIN ----- #

player = Player(Painter())
player.start_player()
