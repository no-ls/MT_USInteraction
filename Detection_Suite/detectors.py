from abc import abstractmethod
import cv2
from cv2.typing import MatLike
from Default_vars import COLORS, KEYS
import numpy as np

NAME_POS = (10, 40)
TEXT_OFFSET = 20

# ----- PARENT ----- #

class Algorithm():
    """Parent class that structures all implemented algorithms"""
    def __init__(self, debug:bool=False) -> None:
        self.debug = debug

    def identify(self):
        print("is", self)
    
    def apply(self, value:int, img:MatLike, gray:MatLike):
        self.write_info(value, img)
        return self.do_algorithm(value, img, gray)

    def write_info(self, value:int, img:MatLike):
        """Write info about the current algorithm on the screen"""
        name = self.__class__.__name__
        text = f"{name} - {value}"
        cv2.putText(img, text, NAME_POS, cv2.FONT_HERSHEY_PLAIN, 1, COLORS.WHITE, 1, cv2.LINE_AA)

    @abstractmethod
    def do_algorithm(self):
        pass


# ----- CHILD CLASSES ---- #

class Threshold(Algorithm):
    """A simple threshold operation for detection"""

    def do_algorithm(self, value:int, img:MatLike, gray:MatLike) -> MatLike:
        """returns the thresholded image"""
        # binary threshold
        _, thresh = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, COLORS.RED, 1)

        return thresh
    
class K_Means(Algorithm):
    """A k-means detector"""

    def do_algorithm(self, value:int, img:MatLike, gray:MatLike) -> MatLike:
        """returns the segmented image"""
        Z = img.reshape((-1,3))
        Z = np.float32(Z)

        # define criteria, number of clusters(K)
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0)
        K = value
        
        # apply kmeans()
        _, label, center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # convert back
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        
        return res2
    

# ----- LIST ----- #

ALGORITHMS = {
    KEYS.ONE: Threshold(),
    KEYS.TWO: K_Means(),
}

DEFAULT_A_VALUES = {
    KEYS.ONE: 50,
    KEYS.TWO: 2,
}
