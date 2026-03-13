import cv2
import time
import open3d as o3d
import numpy as np
from Helpers.Demo_Class import Demo
from Helpers.Player import Player
from Helpers.Parameters import COLORS
from cv2.typing import MatLike

""" 
using: Open3D https://pypi.org/project/open3d/
    -> visualize Point Cloud https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
    -> surface reconstruction (see: https://www.open3d.org/html/tutorial/Advanced/surface_reconstruction.html)
"""
PROBE_ARTIFACT = 30
SCAN_STRIP_WIDTH = 100
DEFAULT_Z_DISTANCE = 1 # default distance between two z values

MAX_COLOR_VALUE = 1.0
MAX_DEPTH_VALUE = 200

SIZE_SCAN_BED = 40000
MIN_SIZE_POST_BED = 30000
BED_DURATION = 20 # how long the scan bed need to be visible for
# Angle at of the diagonal measuring stick /_ 
DIAGONAL_ANGLE = 28.6 # in degrees

SCAN_STICK_THRESHOLD = 70
MIN_GRID_WIDTH = 30
MIN_STICK_CONTOUR_AREA = 200

MIN_SIMILAR_SCORE = 20

class Scanner(Demo):
    def __init__(self) -> None:
        super().__init__()
        self.slider_value = 5
        self.slider_max = 20
        self.pcds = []
        self.i = 1
        self.start_scan = False
        self.is_scan_bed = False
        self.was_scan_bed = False
        self.bed_duration = 0

        self.us_area_threshold = 37

        self.has_init_viz = False
        self.vis = o3d.visualization.Visualizer()
        self.pcd = o3d.geometry.PointCloud()

        self.prev_z = 0
        self.stick_contour = []
        self.prev_max_c = []

        self.similar_score = 0

        self.do_freehand_scan = False

    def start(self):
        self.start_scan = not self.start_scan

    def free_key_interaction(self):
        self.do_freehand_scan = True

    # ---

    def do(self, frame:MatLike, masked:MatLike)-> MatLike:
        super().do(frame, masked)
        # masked = cv2.pyrDown(masked)

        base = masked.copy() # basic, unaltered masked frame

        # mask very top area (probe artifacts)
        cv2.rectangle(masked, (self.us_area.x, self.us_area.y), (self.us_area.x+self.us_area.w, self.us_area.y+PROBE_ARTIFACT), COLORS.BLACK, -1)
        contours, frame = self.segment(masked)

        if not self.start_scan:
            self.write_text(frame, "Start scan with 'ENTER'", (20, 40))
            if self.do_freehand_scan:
                self.write_text(frame, "Do freehand scan", (20, 60))
            else:
                self.write_text(frame, "Press 'f' for freehand scan", (20, 60))
            return frame

        # ----- PRE - SCAN ----- # 

        if len(contours) == 0: return masked
        max_c = max(contours, key=cv2.contourArea)
        cv2.drawContours(masked, [max_c], -1, COLORS.BLUE, 1)

        # ignore the scan bed (should fill out area), if it gets (reliably) detected
        x,y,w,h = cv2.boundingRect(max_c)
        if w > self.us_area.w - 50 and self.is_debug:
            cv2.rectangle(masked,(x,y),(x+w,y+h),COLORS.GREEN,1)
            self.write_text(masked, "ignoring scan bed", (20, 80))
            return masked
        
        # ----- SCAN - PROCESS ----- # 

        # FREEHAND SCAN (i.e. !without! scan diagonals and calculated depth value)
        if self.do_freehand_scan:
            self.write_text(masked, "scanning (freehand)...", (20, 40))
            self.parse_contours(base, contours, self.prev_z, max_c, True) # use unaltered frame
            self.init_viz()
            self.prev_z += 1
            return frame

        # REGULAR SCAN (i.e. using scan diagonal to calculated depth value)

        z = self.get_depth_value(base)

        if z != None:
            self.parse_contours(base, contours, z, max_c) # use unaltered frame
            self.init_viz()
            self.prev_z = z

        # ----- (DEBUG) INFO ----- #
        if self.is_debug:
            if len(max_c) > 0:
                cv2.drawContours(masked, contours, -1, COLORS.RED, 1)
            if len(self.stick_contour) > 0:
                cv2.drawContours(masked, [self.stick_contour], -1, COLORS.GREEN, 1)
        self.write_text(masked, "scanning...", (20, 40))
        self.write_text(masked, f"z = {z}", (20, 60))

        # self.check_contour_similarity(max_c)

        if len(self.prev_max_c) == 0:
            self.prev_max_c = max_c
        return masked
    
    def init_viz(self):
        """Init the real time view of the visualization window (requires initial points), if not yet done"""
        if not self.has_init_viz:
            self.vis.create_window()
            self.vis.add_geometry(self.pcd)
            self.has_init_viz = True

    def get_depth_value(self, masked:MatLike, is_left=True):
        """Map y-coordinates on the left/right side of the image to a depth value"""
        # NOTE: The scan bed has a diagonal measuring stick on the left and right side (going up/down)
                # As the bed is lowered the stick will show up at a different y-value in the US image 

        masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        # HACK-y (adjust if needed) ↓
        side = None
        if is_left:
            # black out the right side, so only the left side is left
            left = masked.copy()
            left = cv2.rectangle(left, (SCAN_STRIP_WIDTH, 0), (self.image_w, self.image_h), COLORS.BLACK, -1) # right side
            left = cv2.rectangle(left, (0, 0), (self.image_w,200), COLORS.BLACK, -1) # top 
            left = cv2.rectangle(left, (0, self.image_h-250), (self.image_w,self.image_h), COLORS.BLACK, -1) # bottom
            side = left
        else: # right + reverse
            right = masked.copy()
            right = cv2.rectangle(right, (0, 0), (self.image_w-SCAN_STRIP_WIDTH, self.image_h), COLORS.BLACK, -1) # left side
            right = cv2.rectangle(right, (self.image_w-200, 0), (self.image_w, 200), COLORS.BLACK, -1) # top 
            right = cv2.rectangle(right, (0, self.image_h-250), (self.image_w,self.image_h), COLORS.BLACK, -1) # bottom
            side = right
    

        # get the contours of the scan sticks
        _, side = cv2.threshold(side, SCAN_STICK_THRESHOLD, 255, cv2.THRESH_BINARY)
        stick_contours, _ = cv2.findContours(side, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(stick_contours) == 0:
            return self.prev_z + DEFAULT_Z_DISTANCE

        # get the biggest contours
        stick_max = max(stick_contours,   key=cv2.contourArea)
        self.stick_contour = stick_max # for showing debug
        stick_area = cv2.contourArea(stick_max)

        cv2.drawContours(masked, [stick_max],-1, COLORS.PURPLE, 1)

        # filter out contours that are too wide, and very small (h) (prop. scan bed)
        sx,sy,sw,sh = cv2.boundingRect(stick_max)
        if sw > sh:
            s_diff = sw - sh
            if s_diff > MIN_GRID_WIDTH:
                print("z: got grid")
                return # drop
            
        # if contours are big enough (to ignore noise) calculate z-value
        z = 0
        if stick_area > MIN_STICK_CONTOUR_AREA:  
            y = self.get_center_Y(stick_max)
            y = self.find_US_y(y, going_down=is_left)
            z = self.calculate_depth(y)
            # print("++ ", z)
            return round(z, 2)
        elif stick_area < MIN_STICK_CONTOUR_AREA:
            if self.prev_z > 0:
                # print("++ default z-value")
                return self.prev_z + DEFAULT_Z_DISTANCE
            else: return # drop

    def get_center_Y(self, contour):
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cy = int(M['m01']/M['m00'])
            return cy
        return 0
    
    def find_US_y(self, y, going_down=True):
        """Subtract the offset of the actual US scan area from the y coordinate"""
        if not going_down: # diagonal coming from back/bottom
            return self.us_area.h - y
        
        return y - self.us_area.y # diagonal coming from front/top

    def calculate_depth(self, y):
        """Use a (irl) y value to calculate the depth is represents (using triangulation)"""
        alpha = DIAGONAL_ANGLE # degrees -> angle of between ground and irl diagonal |↗_|
        beta = 90 # degrees (right angle)
        c = y
        a = 0

        # get remaining angle
        gamma = 180 - alpha - beta
        # use law of sines (sinussatz) to get missing side a
        a = round( ( c / np.sin(np.deg2rad(gamma)) ) * np.sin(np.deg2rad(alpha)), 2 )
        return a 
    
    def contour_is_unchanged(self, max_c:list)->bool:
        """Check if the contour has significantly changed for a while"""
        if len(self.prev_max_c) != 0:
            ret = cv2.matchShapes(self.prev_max_c,max_c,1,0.0) # compare contours
            if ret < 0.1:
                self.similar_score += 1
            else: 
                self.prev_max_c = max_c # update
                self.similar_score = 0
            
            if self.similar_score >= MIN_SIMILAR_SCORE:
                print("contours too similar for too long -> skipping")
                return True
            else: False
    
    def parse_contours(self, masked:MatLike, contours, z, max_c, is_freehand=False):
        """Convert the contours to 3d point clouds and stack them at different heights"""

        # skip if the z-value hasn't changed 
        if self.prev_z == z and not is_freehand:
            return # NOTE: freehand scan uses prev_z here before updating it later on

        # skip scanning once max contour stays the same for a while
        if self.contour_is_unchanged(max_c): return

        # colors for better viewing
        c = np.interp(z, [0, MAX_DEPTH_VALUE], [0, MAX_COLOR_VALUE])
        blue = (0.3, 0.0, c)
        red = (c, 0.0, 0.3)

        if self.i % 2 == 0:
            self.contours_to_3d(contours, z, red)
        else:
            self.contours_to_3d(contours, z, blue)
        self.i += 1

    def contours_to_3d(self, contours, z_value, color:np.array):
        """Turn the OpenCV-contours into a usable format and add them to the Open3D point cloud"""
        if len(contours) == 0: return
        # parse the contours into the right format
        cnts = np.vstack(contours).squeeze(1) 
        pts = cnts.astype(np.float64)
        pts3d = np.hstack([pts, np.full((pts.shape[0], 1), z_value)])

        # add color
        N = pts3d.shape[0]
        colors = np.tile(color, (N, 1))

        # update the point cloud object
        self.pcd.points.extend(o3d.utility.Vector3dVector(pts3d))
        self.pcd.colors.extend(o3d.utility.Vector3dVector(colors)) # float64 (num_points, 3)

        self.update_visualization()

    def update_visualization(self):
        """using non-blocking visualization: https://www.open3d.org/docs/latest/tutorial/visualization/non_blocking_visualization.html"""
        # see for example: https://stackoverflow.com/a/74669788, https://stackoverflow.com/a/78009748
        # NOTE: requires initialization with existing points (see: init_viz())

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def on_finished(self, frame):
        print("[INFO] - Showing stacked point cloud")
        o3d.visualization.draw_geometries([self.pcd])
        # self.save_point_cloud(frame)

    def save_point_cloud(self, frame):  
        print("saving to: ../Data/Models/cloud-{time.time()}.pcd")   
        o3d.io.write_point_cloud(f"../Data/Models/cloud-{time.time()}.pcd", self.pcd)

# ----- MAIN ----- #
# video = "../Data/scan-test-diagonal1.mp4"
# video = "../Data/scan4.mp4"
# video = "../Data/scan3-cut.mp4"
video = "../Data/scan-agar2.mp4"
video = "../Data/nscan-3dball3.mp4"
# video = "../Data/scan-boat.mp4"
player = Player(Scanner(), video)
player.start_player()