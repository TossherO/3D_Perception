import open3d as o3d
import numpy as np
from mmdet3d.visualization import Det3DLocalVisualizer

file_name = 'os1/10/3d_comp_os1_10_1950.bin'
points = np.fromfile('data/CODA/3d_comp/'+file_name, dtype=np.float32).reshape(-1, 4)
points2 = np.fromfile('data/CODA/3d_comp_downsample/'+file_name, dtype=np.float32).reshape(-1, 4)

visualizer1 = Det3DLocalVisualizer()
visualizer2 = Det3DLocalVisualizer()
visualizer1.set_points(points)
visualizer2.set_points(points2)
visualizer1.show()
visualizer2.show()