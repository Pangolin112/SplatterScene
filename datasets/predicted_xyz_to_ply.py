import open3d as o3d
import numpy as np


def convert_txt_to_ply(txt_file_path, ply_file_path):
    # Read the points from the txt file
    points = np.loadtxt(txt_file_path)

    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Write the point cloud to a PLY file
    o3d.io.write_point_cloud(ply_file_path, point_cloud)


file_path = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/xyz_predicted/predicted_xyz550.txt'
ply_path = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/ply_predicted/predicted_ply550.ply'

# Convert the provided text file to PLY
convert_txt_to_ply(file_path, ply_path)
