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
BED_DURATION = 100 # how long the scan bed need to be visible for
# Angle at of the diagonal measuring stick /_ 
DIAGONAL_ANGLE = 28.6 # in degrees

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

        self.has_init_viz = False
        self.vis = o3d.visualization.Visualizer()
        self.pcd = o3d.geometry.PointCloud()

        self.prev_z = 0
        self.left_contour = None
        self.right_contour = None

        self.do_freehand_scan = False

    def start(self):
        self.start_scan = True
    
    def free_key_interaction(self):
        self.do_freehand_scan = True

    def do(self, frame:MatLike, masked:MatLike)-> MatLike:
        super().do(frame, masked)

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
        
        # Option to du freehand scan (i.e. without scan diagonals)
        if self.do_freehand_scan:
            self.write_text(masked, "scanning (freehand)...", (20, 40))
            self.parse_contours(base, contours, self.prev_z) # use unaltered frame
            self.init_viz()
            self.prev_z += 1
            return frame

        # ----- PRE - SCAN ----- # 

        max_c = max(contours, key=cv2.contourArea) 
        area = cv2.contourArea(max_c)

        # check for the scan bed: 
            # if it hasn't been detected yet, the scan does not start
            # if its been detected and disappeared again, then start the scan
        if area >= SIZE_SCAN_BED and not self.was_scan_bed: 
            self.write_text(masked, "Detecting scan bed", (20, 40))
            self.bed_duration += 1
            if self.bed_duration > BED_DURATION:
                self.write_text(masked, "Done!", (20, 60))
                self.is_scan_bed = True
            return masked
        elif area < SIZE_SCAN_BED and self.is_scan_bed:
            self.is_scan_bed = False
            self.was_scan_bed = True
            self.bed_duration = 0
        elif not self.was_scan_bed:
            self.write_text(masked, "Waiting for scan bed", (20, 40))
            return masked

        # ----- SCAN - PROCESS ----- # 

        z = self.get_depth_value(base)

        if z != None:
            self.prev_z = z
            self.parse_contours(base, contours, z) # use unaltered frame
            self.init_viz()
            # if not self.has_init_viz:
            #     self.vis.create_window()
            #     self.vis.add_geometry(self.pcd)
            #     self.has_init_viz = True

        # ----- (DEBUG) INFO ----- #
        if self.is_debug:
            cv2.drawContours(masked, [self.left_contour, self.right_contour], -1, COLORS.GREEN, 1)
        self.write_text(masked, "scanning...", (20, 40))
        self.write_text(masked, f"z = {z}", (20, 60))

        return masked
    
    def init_viz(self):
        """init the real time view of the visualization window (requires initial points), if not yet done"""
        if not self.has_init_viz:
            self.vis.create_window()
            self.vis.add_geometry(self.pcd)
            self.has_init_viz = True

    
    def get_depth_value(self, masked:MatLike):
        """Map y-coordinates on the left/right side of the image to a depth value"""
        # NOTE: The scan bed has a diagonal measuring stick on the left and right side (going up/down)
                # As the bed is lowered the stick will show up at a different y-value in the US image 

        masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        # black out the right side
        left = masked.copy()
        left = cv2.rectangle(left, (SCAN_STRIP_WIDTH, 0), (self.image_w, self.image_h), COLORS.BLACK, -1) # right side
        # left = cv2.rectangle(left, (0, 0), (self.image_w,200), COLORS.BLACK, -1) # top 
        left = cv2.rectangle(left, (0, self.image_h-150), (self.image_w,self.image_h), COLORS.BLACK, -1) # bottom

        # black out the left side
        right = masked.copy()
        right = cv2.rectangle(right, (0, 0), (self.image_w-SCAN_STRIP_WIDTH, self.image_h), COLORS.BLACK, -1) # left side
        right = cv2.rectangle(right, (0, self.image_h-150), (self.image_w,self.image_h), COLORS.BLACK, -1) # bottom

        # get the contours of the scan sticks
        _, left = cv2.threshold(left, 70, 255, cv2.THRESH_BINARY)
        left_contours, _ = cv2.findContours(left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        _, right = cv2.threshold(right, 70, 255, cv2.THRESH_BINARY)
        right_contours, _ = cv2.findContours(right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # get the biggest contours
        left_max = max(left_contours,   key=cv2.contourArea)
        right_max = max(right_contours, key=cv2.contourArea) 
        self.left_contour = left_max # for showing debug
        self.right_contour = right_max 

        left_area = cv2.contourArea(left_max)
        right_area = cv2.contourArea(right_max)

        cv2.drawContours(masked, [left_max],-1, COLORS.PURPLE, 1)
        cv2.drawContours(masked, [right_max],-1, COLORS.GREEN, 1)

        # if contours are big enough (ignore noise) calculate z-value
        left_z = 0
        right_z = 0
        if left_area > 700 :
            left_y = self.get_center_Y(left_max)
            left_y = self.find_US_y(left_y)
            left_z = self.calculate_depth(left_y)
        if right_area > 700 :
            right_y = self.get_center_Y(right_max)
            right_y = self.find_US_y(right_y, inverse=True)
            right_z = self.calculate_depth(right_y)
        if left_area < 700 and right_area < 700: # update a previous value or drop
            if self.prev_z > 0: 
                return self.prev_z + DEFAULT_Z_DISTANCE
            else: return # drop

        # return the new depth value            
        if left_z != 0 and right_z != 0:
            return round(np.average([left_z, right_z]),2)
        elif left_z != 0 and right_z == 0: 
            return left_z
        else:
            return right_z

    def get_center_Y(self, contour):
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cy = int(M['m01']/M['m00'])
            return cy
        return 0
    
    def find_US_y(self, y, inverse=False):
        """Subtract the offset of the actual US scan area from the y coordinate"""
        if inverse: # diagonal coming from back/bottom
            return self.us_area.h - y
        # diagonal coming from front/top
        return y - self.us_area.y

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
    
    def parse_contours(self, masked:MatLike, contours, z):
        """Convert the contours to 3d point clouds and stack them at different heights"""
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
        # NOTE: requires initialization with existing points

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def on_finished(self, frame):
        if not self.started:
            print("No scan :(")
            return
        print("[INFO] - Showing stacked point cloud")
        o3d.visualization.draw_geometries([self.pcd])

    def save(self, frame):  
        print("saving to: ../Data/Models/cloud-{time.time()}.pcd")   
        o3d.io.write_point_cloud(f"../Data/Models/cloud-{time.time()}.pcd", self.pcd)

# ----- MAIN ----- #
# video = "../Data/scan-test-diagonal1.mp4"
video = "../Data/scan4.mp4"
player = Player(Scanner(), video)
player.start_player()