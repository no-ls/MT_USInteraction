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

    def do_threshold(self, gray, value):
        _, thresh = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
        return thresh
    
    def do_adaptive_threshold(self, gray, value):
        # TODO try preblur + changing vars
        gray = cv2.GaussianBlur(gray, (5,15), 0)
        blockSize = 11
        C = 2
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,blockSize,C)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,blockSize,C)
        return thresh
    
    def do_Otsu(self, gray, value):
        # blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh

    def do_algorithm(self, value:int, img:MatLike, gray:MatLike) -> MatLike:
        """returns the thresholded image"""

        # THRESHOLDS        
        thresh = self.do_threshold(gray, value)
        # thresh = self.do_adaptive_threshold(gray, value)
        # thresh = self.do_Otsu(gray, value)

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
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

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
    
class Brightest_Spot(Algorithm):
    def do_algorithm(self, value:int, img:MatLike, gray:MatLike):
        img = cv2.GaussianBlur(img, (7,7), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        cv2.circle(img, maxLoc, 5, COLORS.BLUE, 2)
        cv2.putText(img, " brightest", maxLoc, 1, 1, COLORS.BLUE, 1)

        _, thresh = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(thresh)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        cv2.circle(img, (cx, cy), 5, COLORS.RED, 2)
        cv2.putText(img, " moment", (cx, cy), 1, 1, COLORS.RED, 1)

        return img
    
class Optical_Flow(Algorithm):
    "via: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html"
    def __init__(self, debug: bool = False) -> None:
        super().__init__(debug)
        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
        self.prev = []
        self.color = np.random.randint(0, 255, (100, 3))
        self.p0 = None

    def do_algorithm(self, value:int, frame:MatLike, gray:MatLike):
        """Only works with manual switch right now (28.10.25)"""

        lk = False

        if not lk:
        # do OPTICAL FLOW (full)
            if len(self.prev) == 0:
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                self.prev = gray
                self.hsv = np.zeros_like(frame)
                self.hsv[..., 1] = 255
                return frame
            next = gray
            flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            self.hsv[..., 0] = ang*180/np.pi/2
            self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            out = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
            out = cv2.add(frame, out)

        # do OPTICAL FLOW (LK)
        else:
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if len(self.prev) == 0:
                self.prev = gray

                # get brightest spot
                img = cv2.GaussianBlur(frame, (7,7), 0)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
                x, y = maxLoc
                brightest = [[x, y]]
                self.p0 =  np.array([brightest], dtype=np.float32) # to proper datatype
                # self.p0 = cv2.goodFeaturesToTrack(self.prev, mask=None, **self.feature_params)
                self.mask = np.zeros_like(self.prev)
                return frame
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.p0, None, **self.lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[st==1]
                good_old = self.p0[st==1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2) # lines
                out = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)

        # out = frame
        # self.show_fps(out)
        return out

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

class Pre_Processing(Algorithm):
    """Several preprocessing techniques"""

    def write(self, img, text):
        pos = (300, 40)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_PLAIN, 1, COLORS.WHITE, 1, cv2.LINE_AA)

    def smooth(self, img): # lowpass
        kernel = np.ones((5,5), np.float32) / 25
        img = cv2.filter2D(img, -1, kernel)
        self.write(img, "smooth (filter2D)")
        return img

    def dilate(self, img):
        kernel = np.ones((5,5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        self.write(img, "dilate")
        return img

    def erode(self, img):
        kernel = np.ones((5,5), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        self.write(img, "erode")
        return img

    def opening(self, img):
        kernel = np.ones((5,5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        self.write(img, "opening")
        return img        

    def closing(self, img):
        kernel = np.ones((5,5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)        
        self.write(img, "closing")
        return img

    def downsample(self, img):
        resample = 2
        rows, cols, _channels = map(int, img.shape)
        img = cv2.pyrDown(img, dstsize=(cols // resample, rows // resample))
        
        # upsample for display
        rows, cols, _channels = map(int, img.shape)
        img = cv2.pyrUp(img, dstsize=(resample * cols, resample * rows))
        self.write(img, "down sample")        
        return img
    
    def highpass(self, img):
        """via: https://pythonexamples.org/python-opencv-image-filter-convolution-cv2-filter2d/"""
        kernel = np.array([[0.0, -1.0, 0.0], 
                   [-1.0, 5.0, -1.0],
                   [0.0, -1.0, 0.0]])
        kernel = kernel/(np.sum(kernel) if np.sum(kernel) != 0 else 1)
        img = cv2.filter2D(img, -1, kernel)
        self.write(img, "highpass")        
        return img

    def denoise(self, img):
        """variable explanation: https://stackoverflow.com/a/37921901"""
        h = 20 # odd, higher = less detail less noise
        templateWindowSize = 7 # odd
        searchWindowSize = 5 # higher value affects time
        img = cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)
        self.write(img, "non-local means denoise")            
        return img
    
    def do_algorithm(self, value:int, img:MatLike, gray:MatLike):
        kernel = (5, 5)

        match value:
            case 0:
                img = cv2.blur(img, kernel)
                self.write(img, "blur")
            case 1:
                img = cv2.GaussianBlur(img, kernel, 0)
                self.write(img, "GaussianBlur")
            case 2:
                img = cv2.medianBlur(img, 5)
                self.write(img, "medianBlur")
            case 3:
                img = cv2.bilateralFilter(img, 9, 75, 75)
                self.write(img, "bilateralFilter")
            case 4:
                img = self.smooth(img)
            case 5:
                img = self.dilate(img)
            case 6:
                img = self.erode(img)
            case 7:
                img = self.closing(img)
            case 8:
                img = self.opening(img)
            case 9:
                img = self.downsample(img)
            case 10: 
                img = self.highpass(img)
            case 11:
                img = self.denoise(gray)
            case _:
                self.write(img, "none")
        
        return img

# ----- LIST ----- #

ALGORITHMS = {
    KEYS.ONE: Threshold(),
    KEYS.TWO: K_Means(),
    KEYS.THREE: Color_Quantization(),
    KEYS.FOUR: LUT(),
    KEYS.FIVE: Pyr_Mean_Shift(),
    KEYS.SIX: Brightest_Spot(),
    KEYS.SEVEN: Watershed(),
    KEYS.EIGHT: Optical_Flow(),
    KEYS.NINE: Pre_Processing()
}

DEFAULT_A_VALUES = {
    KEYS.ONE: 50,
    KEYS.TWO: 2,
    KEYS.THREE: 2,
    KEYS.FOUR: 3,
    KEYS.FIVE: 30,
    KEYS.SIX: 50,
    KEYS.SEVEN: 0,
    KEYS.EIGHT: 1,
    KEYS.NINE: 0,
}
