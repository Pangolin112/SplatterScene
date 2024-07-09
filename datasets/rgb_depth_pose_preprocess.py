from PIL import Image
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# Define the source and destination folders
scene_name = '355e5e32db'

# rgb
source_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/{scene_name}/images'
destination_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/{scene_name}/rgb'

for file_name in os.listdir(source_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Check for image files
        # Open the image
        img_path = os.path.join(source_folder, file_name)
        img = Image.open(img_path)

        # Convert the image to 'F' mode (32-bit floating point) for resizing
        # img = img.convert('F')

        # Resize the image
        img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)

        # Save the resized image to the destination folder as JPEG
        jpeg_file_name = os.path.splitext(file_name)[0] + '.JPG'
        img_resized.save(os.path.join(destination_folder, jpeg_file_name), format='JPEG')

print("RGB Images resized and saved successfully.")

# pose
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

            # Create the 4x4 world to camera transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rot_matrix
            transform_matrix[:3, 3] = [tx, ty, tz]

            # Store in dictionary
            image_poses[image_name] = np.linalg.inv(transform_matrix)  # store camera to world poses

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
images_txt_path = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/{scene_name}/colmap/images.txt'
output_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/{scene_name}/pose_colmap_depth/'
image_poses = parse_images_txt(images_txt_path)
write_matrices(image_poses, output_folder)

print("Pose extracted and saved successfully.")

# depth
source_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/{scene_name}/render_depth'
destination_folder = f'/media/qianru/12T_Data/Data/ScanNetpp/data_1/{scene_name}/depth'

for file_name in os.listdir(source_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Check for image files
        # Open the image
        img_path = os.path.join(source_folder, file_name)
        img = Image.open(img_path)

        # Convert the image to 'F' mode (32-bit floating point) for resizing
        img = img.convert('F')

        # Resize the image
        img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)

        # Convert to numpy array for scaling
        img_array = np.array(img_resized)

        img_array[img_array < 0] = 0  # threshold for outliers

        # Convert the numpy array back to a PIL Image in 16-bit mode
        result_img = Image.fromarray(img_array.astype(np.uint16))

        # Save the image to the destination folder in 16-bit PNG format
        png_file_name = os.path.splitext(file_name)[0] + '.png'
        result_img.save(os.path.join(destination_folder, png_file_name), format='PNG')

print("Depth resized and saved successfully.")