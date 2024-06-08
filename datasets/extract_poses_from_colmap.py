import os
import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_images_txt(images_txt_path):
    with open(images_txt_path, 'r') as file:
        lines = file.readlines()

    # Dictionary to hold image data
    image_poses = {}

    # Skip the header lines and empty lines
    for line in lines:
        if line.startswith('#') or len(line.strip()) == 0:
            continue

        parts = line.split()
        if len(parts) == 10:  # This is the line with pose data
            image_id = parts[0]
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            image_name = parts[9]

            # Convert quaternion to rotation matrix
            rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

            # Create the 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rot_matrix
            transform_matrix[:3, 3] = [tx, ty, tz]

            # Store in dictionary
            image_poses[image_name] = transform_matrix

    return image_poses


def write_matrices(image_poses, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for image_name, matrix in image_poses.items():
        output_file_path = os.path.join(output_folder, f"{image_name}.txt")
        with open(output_file_path, 'w') as file:
            #for row in matrix:
                #file.write(' '.join(f"{val:g}" for val in row) + '\n')
            matrix_flat = matrix.flatten()  # Flatten the matrix to a 1D array
            file.write(' '.join(f"{val:g}" for val in matrix_flat) + '\n')  # Write all elements in one line


# Usage
images_txt_path = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/images.txt'
output_folder = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/pose_colmap/'
image_poses = parse_images_txt(images_txt_path)
write_matrices(image_poses, output_folder)
