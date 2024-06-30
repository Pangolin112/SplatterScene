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


def add_to_visualizer(directory_path, output_ply_path, world_to_camera=True, transformation_matrix=np.eye(4)):
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter only the relevant files
    pose_files = [file for file in files if file.endswith('.txt')]

    geometries = []

    # Read the transformation matrices from the files and add them to the visualizer
    for pose_file in pose_files:
        file_path = os.path.join(directory_path, pose_file)
        transformation = read_transformation_from_file(file_path)

        if world_to_camera:
            # Invert the transformation matrix if it represents world-to-camera transformation
            transformation = np.linalg.inv(transformation)

        # Apply the given transformation matrix (to switch coordinate systems)
        transformation = transformation_matrix @ transformation

        # Create coordinate frame for the axes
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        coordinate_frame.transform(transformation)

        # Create frustum for the camera
        frustum = create_camera_frustum(size=0.5)
        frustum.transform(transformation)

        # Add both to the visualizer
        # vis.add_geometry(coordinate_frame)
        # vis.add_geometry(frustum)

        # Add both to the geometries list
        geometries.append(coordinate_frame)
        geometries.append(frustum)

        # Combine all geometries into a single mesh and line set
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_line_set = o3d.geometry.LineSet()

    for geometry in geometries:
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            combined_mesh += geometry
        elif isinstance(geometry, o3d.geometry.LineSet):
            combined_line_set += geometry

        # Save the combined geometries to a .ply file
    o3d.io.write_triangle_mesh(output_ply_path, combined_mesh)
    o3d.io.write_line_set(output_ply_path.replace(".ply", "_lines.ply"), combined_line_set)


# Directory containing the transformation files
scene_0cf2e9402d_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/srn_cars/cars_train/0cf2e9402d_train/pose_colmap_depth/'
#scene_0cf2e9402d_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/0cf2e9402d/pose_nerfstudio_rgb/'
scene_0cf2e9402d_out_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/camera_ply_outpath/0cf2e9402d_w2c.ply'
scene_49a82360aa_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1//49a82360aa/pose_colmap_depth/'
scene_49a82360aa_out_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/camera_ply_outpath/49a82360aa.ply'
srn_path = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/SRN/srn_cars/cars_train/1a1dcd236a1e6133860800e6696b8284/pose/'
srn_out_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/camera_ply_outpath/srn.ply'
#directory_path = '/media/qianru/12T_Data/Data/ScanNetpp/data_1/49a82360aa/pose_nerfstudio_rgb/'
# Create a visualizer
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# COLMAP to Open3D transformation matrix
T_COLMAP_to_Open3D = np.array([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])

# ScanNet++ (World to Camera) to Open3D transformation matrix
T_ScanNet_to_Open3D = np.array([[1, 0, 0, 0],
                                [0, 0, -1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]])

# SRN (Camera to World) to Open3D transformation matrix (inverse of ScanNet++ to Open3D)
T_SRN_to_Open3D = np.linalg.inv(T_ScanNet_to_Open3D)

add_to_visualizer(scene_0cf2e9402d_path, scene_0cf2e9402d_out_path, True)
add_to_visualizer(scene_49a82360aa_path, scene_49a82360aa_out_path, False)
add_to_visualizer(srn_path, srn_out_path, True)

# Render the scene
# vis.run()
# vis.destroy_window()

