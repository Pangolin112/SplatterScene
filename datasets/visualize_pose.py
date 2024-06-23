import numpy as np
import open3d as o3d
import os


# Function to read transformation matrix from a file
def read_transformation_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = []
        for line in lines:
            matrix.extend(map(float, line.split()))
        transformation_matrix = np.array(matrix).reshape(4, 4)
    return transformation_matrix


# Function to create a camera frustum
def create_camera_frustum(size=0.01, color=[0, 0, 1]):
    points = [
        [0, 0, 0],  # Camera origin
        [size, -0.5 * size, 0.75 * size],  # Top-right
        [size, 0.5 * size, 0.75 * size],  # Top-left
        [size, -0.5 * size, -0.75 * size],  # Bottom-right
        [size, 0.5 * size, -0.75 * size],  # Bottom-left
    ]
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from camera origin to frustum corners
        [1, 2], [2, 4], [4, 3], [3, 1]  # Frustum edges
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


# Directory containing the transformation files
directory_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/cars_train/0cf2e9402d_train/pose/'

#directory_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/0cf2e9402d/pose_colmap/'

directory_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/0cf2e9402d/pose_nerfstudio/'

# List all files in the directory
files = os.listdir(directory_path)

# Filter only the relevant files
pose_files = [file for file in files if file.endswith('.txt')]

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Read the transformation matrices from the files and add them to the visualizer
for pose_file in pose_files:
    file_path = os.path.join(directory_path, pose_file)
    transformation = read_transformation_from_file(file_path)

    # Create coordinate frame for the axes
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    coordinate_frame.transform(transformation)

    # Create frustum for the camera
    frustum = create_camera_frustum(size=0.5)
    frustum.transform(transformation)

    # Add both to the visualizer
    vis.add_geometry(coordinate_frame)
    vis.add_geometry(frustum)

# Render the scene
vis.run()
vis.destroy_window()
