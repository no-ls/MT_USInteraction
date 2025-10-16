from abc import abstractmethod
import cv2
from cv2.typing import MatLike
from Default_vars import COLORS, KEYS
import numpy as np

# ----- PARENT ----- #

class Algorithm():
    """Parent class that structures all implemented algorithms"""
    def __init__(self, debug:bool=False) -> None:
        self.debug = debug

    def identify(self):
        print("is", self)
    
    def apply(self, value:int, img:MatLike, gray:MatLike):
        return self.do_algorithm(value, img, gray)

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
        # print(value)

        return thresh
    
class K_Means(Algorithm):
    """FOR TESTING KEYBOARD INTERACTION"""

    def do_algorithm(self, value:int, img:MatLike, gray:MatLike) -> MatLike:
        """returns the segmented image"""
        Z = img.reshape((-1,3))
        Z = np.float32(Z)

        # define criteria, number of clusters(K)
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0)
        K = value
        
        # apply kmeans()
        ret, label, center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

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
