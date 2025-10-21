from abc import abstractmethod
from cv2.typing import MatLike

class Demo():
    def __init__(self) -> None:
        pass

    def get_name(self):
        return self.__class__.__name__
    
    @abstractmethod
    def do(self, frame:MatLike, gray:MatLike)-> MatLike:
        """Abstract Method to be overridden by the specific demos."""
        pass
