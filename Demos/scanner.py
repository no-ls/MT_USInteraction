import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from mpl_toolkits import mplot3d
from Helpers.Demo_Class import Demo
from Helpers.Player import Player
from Helpers.Parameters import COLORS
from cv2.typing import MatLike

# TODO "3D" reconstruct from image
""" Options:
    - [x] numpy-stl: https://pypi.org/project/numpy-stl/ -> can view with matplotlib
        - works with vertices and faces -> would need to convert a "point cloud" to triangles
    # pymesh: https://pymesh.readthedocs.io/en/latest/
    # [ ] Open3D ? https://pypi.org/project/open3d/
        - visualize Point Cloud https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
    - pyvista https://pypi.org/project/pyvista/ (API for VTK)
    """

class Scanner(Demo):
    def __init__(self) -> None:
        super().__init__()
        self.slider_value = 5
        self.slider_max = 20
        self.pcds = []
        self.i = 1

    def do(self, frame:MatLike, masked:MatLike)-> MatLike:
        super().do(frame, masked)

        # extract contours and coordinates
        contours, frame = self.segment(masked)

        # NOTE: simulates scan -> only call when "video ends"
        # stack a couple of contours at different z-levels (+ colors), then visualize
        if self.i < 10:
            r = 1.0 / self.i
            self.contours_to_3d(contours, self.i+2, (r, 0.0, 0.0))
            self.i += 1
        else:
            o3d.visualization.draw_geometries(self.pcds)
        
        return frame

    def contours_to_3d(self, contours, z_value, color:np.array):
        # parse th contours into the right format
        cnts = np.vstack(contours).squeeze(1) 
        pts = cnts.astype(np.float64)
        pts3d = np.hstack([pts, np.full((pts.shape[0], 1), z_value)])

        # add color
        N = pts3d.shape[0]
        colors = np.tile(color, (N, 1))

        # create a point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3d)
        pcd.colors = o3d.utility.Vector3dVector(colors) # float64 (num_points, 3)

        # save for later
        self.pcds.append(pcd)

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
video = "../Data/stress.png"
# video = "../Data/stressball.mp4"
player = Player(Scanner(), video)
player.start_player()