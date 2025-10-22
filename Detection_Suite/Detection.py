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
    def do_algorithm(self, value:int, img:MatLike, gray:MatLike):
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
    """A k-means detector.
    via: https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    """

    def do_algorithm(self, value:int, img:MatLike, gray:MatLike) -> MatLike:
        """returns the segmented image. value = Ks"""
        
        # downsample frame for faster computation
        resample = 2
        rows, cols, _channels = map(int, img.shape)
        img = cv2.pyrDown(img, dstsize=(cols // resample, rows // resample))

        # reduce colors (kinda helps) via: https://stackoverflow.com/a/20715062
        # div = 64 / 2 
        # img = img // div * div + div // 2

        # blur
        cv2.GaussianBlur(img, (5,5), 0) 

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

        # upsample for display
        rows, cols, _channels = map(int, res2.shape)
        res2 = cv2.pyrUp(res2, dstsize=(resample * cols, resample * rows))
        
        return res2
# maybe alternative: https://stackoverflow.com/questions/49710006/fast-color-quantization-in-opencv
    
class Watershed(Algorithm):
    """via: dev.to/jarvissan22/python-cv2-image-segmentation-canny-edges-watershed-and-k-means-methods-18l0"""

    def do_algorithm(self, value:int, img:MatLike, gray:MatLike):
        # binary image inversion, this  ensures that foreground objects are white, and the background is black.
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV  | cv2.THRESH_OTSU)

        # Noise removal using morphological opening, this reduced segmentation artifacts 
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Distance transform to highlight object centers, this indicated distance from nearest pixel boundary 
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        #  Thresholding identifies "sure foreground" regions, what is the areas most likely part of objects.
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        # Identify the sure background and unknown regions
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Create markers for watershed segmentation
        markers = cv2.connectedComponents(sure_fg.astype(np.uint8))[1]
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply Watershed, Pixels marked -1 represent boundary lines, which are visually highlighted.
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [255, 255, 255]  # Mark boundaries in red
        return opening

class Color_Quantization(Algorithm):
    """via: https://stackoverflow.com/a/66339640 """
    def do_algorithm(self, value:int, img:MatLike, gray:MatLike):
        # convert to gray as float in range 0 to 1
        num_colors = value
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)/255

        # quantize and convert back to range 0 to 255 as 8-bits
        result = 255*np.floor(gray*num_colors+0.5)/num_colors
        result = result.clip(0,255).astype(np.uint8)

        # find contours
        _, thresh = cv2.threshold(result, 75, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, COLORS.RED, 1)
        return result

class LUT(Algorithm):
    """via: https://github.com/PacktPublishing/Hands-On-Algorithms-for-Computer-Vision/blob/master/Chapter03/CvLutPy/CvLutPy.py"""
    def do_algorithm(self, value:int, img:MatLike, gray:MatLike):
        lut = np.zeros((1, 256), np.uint8)
        for i in range(0, 255):
            if i < value:
                lut[0, i] = 0
            elif i > 130:
                lut[0, i] = 255
            else :
                lut[0, i] = i

        result = cv2.LUT(img, lut)
        return result
    
class Pyr_Mean_Shift(Algorithm):
    def do_algorithm(self, value:int, img:MatLike, gray:MatLike):
        img = cv2.pyrMeanShiftFiltering(img,sp=5, sr=30)
        return img

# doesn't work
class Graph_Cut(Algorithm):
    """via: https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html"""

    def do_algorithm(self, value:int, img:MatLike, gray:MatLike):
        mask = np.zeros(img.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        rect = (150, 150, 300, 300) # TODO find
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype("uint8")
        img = img*mask2[:,:,np.newaxis]
        return img
        # test with "manually" selected box


# ----- LIST ----- #

ALGORITHMS = {
    KEYS.ONE: Threshold(),
    KEYS.TWO: K_Means(),
    KEYS.THREE: Color_Quantization(),
    KEYS.FOUR: LUT(),
    KEYS.FIVE: Pyr_Mean_Shift(),
    KEYS.SIX: Watershed(),
}

DEFAULT_A_VALUES = {
    KEYS.ONE: 50,
    KEYS.TWO: 2,
    KEYS.THREE: 2,
    KEYS.FOUR: 80,
    KEYS.FIVE: 30,
    KEYS.SIX: 0,
    KEYS.SEVEN: 0,
}
