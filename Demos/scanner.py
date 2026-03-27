import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from mpl_toolkits import mplot3d
from Helpers.Demo_Class import Demo
from Helpers.Player import Player
from Helpers.Parameters import COLORS, KEYS
from cv2.typing import MatLike

""" 
using: Open3D https://pypi.org/project/open3d/
    -> visualize Point Cloud https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
    -> surface reconstruction (see: https://www.open3d.org/html/tutorial/Advanced/surface_reconstruction.html)
"""
PROBE_ARTIFACT = 30

class Scanner(Demo):
    def __init__(self) -> None:
        super().__init__()
        self.slider_value = 5
        self.slider_max = 20
        self.pcds = []
        self.i = 1
        self.started = False

        self.vis = o3d.visualization.Visualizer()
        self.pcd = o3d.geometry.PointCloud()

        self.vis.create_window()
        self.vis.add_geometry(self.pcd)

    def do(self, frame:MatLike, masked:MatLike)-> MatLike:
        super().do(frame, masked)


        # TODO: get contours of left side get coordinate OF CENTER
        # TODO: map y-pos(? aka height) to z-value of slice

        # HACK: zum (sehr!) groben ausprobieren

        # black out everything that isn't the left side (or is annoying)
        depth = masked.copy()
        depth = cv2.rectangle(depth, (100, 0), (self.image_w,self.image_h), COLORS.BLACK, -1) # right side
        depth = cv2.rectangle(depth, (0, 0), (self.image_w,200), COLORS.BLACK, -1) # top 
        depth = cv2.rectangle(depth, (0, self.image_h-200), (self.image_w,self.image_h), COLORS.BLACK, -1) # bottom

        # find correct contour (sehr unflexibel)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        _, depth = cv2.threshold(depth, 70, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(depth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        max_c = max(contours, key = cv2.contourArea)
        max_area = cv2.contourArea(max_c)
        if max_area > 700:
            cv2.drawContours(depth, [max_c], -1, color=COLORS.RED, thickness=1)

            # get y coordinate
            M = cv2.moments(max_c)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.circle(depth, (cx, cy), 7, (0, 0, 255), -1)

                # PARSE y coordinate to depth
                real_y = self.parse_y_to_mm(cy)
                depth_value = self.calculate_depth(real_y)
                frame = self.scan(masked, depth_value)    


        # frame = depth # for out
        # contours, frame = self.segment(masked)
        # cv2.drawContours(frame, contours, -1, color=COLORS.RED, thickness=1)

        # self.is_debug = True

        # TODO redo (blocks performance)
        """
        # start the scanning process manually
        key = cv2.waitKeyEx(25) # bad for performance
        if key == KEYS.ENTER:
            self.started = True
            
            # start the non-blocking visualization
            self.vis.create_window()
            self.vis.add_geometry(self.pcd)

        # self.started = True # for testing

        if not self.started:
            self.write_text(frame, "Start scan with 'ENTER'", (20, 20))
            return frame
        """
            
        # frame = self.scan(masked)    
        # self.write_text(frame, "scanning...", (20, 20))

        return frame
    
    def parse_y_to_mm(self, y):
        """Takes a y-coordinate inside the US scan area and interpolates it to the real life position (in mm)"""
        # x1 = 180, x15 = 130 # mm
        real_h = 130 # mm

        # find position in US_AREA
        us_y = y - self.us_area.y
        return us_y 
        # NOTE: keep it in image scale to keep same proportions as contours

        # map y coordinate to mm
        real_y = round(np.interp(us_y, [0, self.us_area.h], [0, real_h]))

        # print(us_y, "->", real_y, "in mm")
        return real_y

    def calculate_depth(self, y):
        """Use a (irl) y value to calculate the depth is represents (using triangulation)"""
        alpha = 18 # degrees -> angle of between ground and irl diagonal |↗_|
        beta = 90 # degrees (right angle)
        c = y
        a = 0 # result in ...?

        # get remaining angle
        gamma = 180 - alpha - beta
        # use law of sines (sinussatz) to get missing side a
        a = round( ( c / np.sin(np.deg2rad(gamma)) ) * np.sin(np.deg2rad(alpha)), 2 )
        
        # normalize the value?
        # print("Depth =",  a, "mm")
        return a 
    
    def scan(self, masked:MatLike, z):
        """Capsules the scanning behavior"""

        # mask very top
        cv2.rectangle(masked, (self.us_area.x, self.us_area.y), (self.us_area.x+self.us_area.w, self.us_area.y+PROBE_ARTIFACT), COLORS.BLACK, -1)

        masked = cv2.pyrDown(masked)

        # extract contours and coordinates
        contours, frame = self.segment(masked)
        cv2.drawContours(frame, contours, -1, color=COLORS.RED, thickness=1)

        frame = cv2.pyrUp(frame)

        # TODO: fix z-values -> try diagonal thingy
        # TODO: update z-values with input: height
        # TODO: scale point cloud to size

        # move probe -> prob. many motion artifacts + difficult
            # man kann eine z-Achsen Orientierung im Becken anbringen
            # z.B. "Stab" diagonal "\" vorne/hinten anbringen um dann "position" in 2D cross section = tiefe
        # move object -> prop. easier to do, but harder to implement
            # but: how to tell z-indexes -> how big an object is
            # z-index am Ende anpassen -> z.B.: manuelle Eingabe, wie hoch Objekt ist
            # z.B.: mit Gehäuse (ähnlich wie oben)
        # -> both options -> wenns kein Anhaltspunkt, dann kann man manuell eingeben oder default z-wert
        
        # stack a couple of contours at different z-levels (+ colors), then visualize
        r = 1.0 / self.i * 10
        # self.contours_to_3d(contours, self.i+2, (r, 0.0, 0.0))

        # colors for better viewing
        red = (1.0, 0.0, 0.0)
        blue = (0.0, 0.0, 1.0)
        if self.i % 2 == 0:
            self.contours_to_3d(contours, z, red)
            # self.contours_to_3d(contours, self.i+2, red)
        else:
            self.contours_to_3d(contours, z, blue)
            # self.contours_to_3d(contours, self.i+2, blue)
        self.i += 1

        return frame

    def contours_to_3d(self, contours, z_value, color:np.array):
        if len(contours) == 0: return
        # parse th contours into the right format
        cnts = np.vstack(contours).squeeze(1) 
        pts = cnts.astype(np.float64)
        pts3d = np.hstack([pts, np.full((pts.shape[0], 1), z_value)])

        # add color
        N = pts3d.shape[0]
        colors = np.tile(color, (N, 1))

        # update the point cloud object
        self.pcd.points.extend(o3d.utility.Vector3dVector(pts3d))
        self.pcd.colors.extend(o3d.utility.Vector3dVector(colors)) # float64 (num_points, 3)

        # self.update_visualization() # no work right now :/ (13.02)

    def update_visualization(self):
        """using non-blocking visualization: https://www.open3d.org/docs/latest/tutorial/visualization/non_blocking_visualization.html"""
        # see for example: https://stackoverflow.com/a/74669788, https://stackoverflow.com/a/78009748

        # print("pcd", self.pcd)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def on_finished(self, frame):
        # if not self.started:
            # print("No scan :(")
            # return
        print("[INFO] - Showing stacked point cloud")
        o3d.visualization.draw_geometries([self.pcd])
        # self.save()

    def save(self):
        # https://www.open3d.org/docs/release/tutorial/geometry/file_io.html#Point-cloud
        # combine the point clouds
        pcd = self.pcds[0]
        i = 0
        for cloud in self.pcds:
            if i == 0: i +=1  # skip first
            else:
                pcd = pcd + cloud
        # o3d.io.write_point_cloud("../Data/Models/cloud.pcd", pcd)

    # NOTE: simple (static) working example -> rm later
    def show_point_cloud(self, contours, z_value):
        """visualizes the found contours as a point could by using the o3d viewer"""

        cnts = np.vstack(contours).squeeze(1) 
        pts = cnts.astype(np.float64)
        pts3d = np.hstack([pts, np.full((pts.shape[0], 1), z_value)])
        color = np.array((1.0, 0.0, 0.0)) # red
        N = pts3d.shape[0]
        colors = np.tile(color, (N, 1))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3d)
        pcd.colors = o3d.utility.Vector3dVector(colors) # float64 (num_points, 3)
        o3d.visualization.draw_geometries([pcd])


# NOTE: visualize a single slice with plt see: https://stackoverflow.com/questions/76626930/how-do-i-give-the-contours-read-in-opencv-to-matplotlib-for-display

# ----- MAIN ----- #
# video = "../Data/scan1.mp4"
video = "../Data/scan-test-diagonal1.mp4"
player = Player(Scanner(), video)
player.start_player()