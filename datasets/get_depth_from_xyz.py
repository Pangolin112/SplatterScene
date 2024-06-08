import numpy as np
import matplotlib.pyplot as plt
import cv2


def read_intrinsics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        fx, cx, cy, _ = map(float, lines[0].split())
        _ = list(map(float, lines[1].split()))  # ignore this line as it is usually zeroes
        _ = list(map(float, lines[2].split()))  # ignore this line as it is usually ones
        width, height = map(int, lines[3].split())
    K = np.array([[fx, 0, cx],
                  [0, fx, cy],
                  [0, 0, 1]])
    return K, width, height


def read_extrinsics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        extrinsics = np.array([list(map(float, line.split())) for line in lines]).reshape((4, 4))
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    return R, t


def read_points(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # skip the header line
        points = np.array([list(map(float, line.split())) for line in lines])
    return points


def project_points_to_image_plane(points, K, R, t, extrinsics_direction='world_to_camera'):
    """
    Projects 3D points onto the 2D image plane and computes their depth.

    Args:
        points (np.ndarray): Nx3 array of 3D points in world coordinates.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        R (np.ndarray): 3x3 rotation matrix (extrinsic parameter).
        t (np.ndarray): 3x1 translation vector (extrinsic parameter).

    Returns:
        projected_points (np.ndarray): Nx2 array of 2D points in image coordinates.
        depths (np.ndarray): N array of depth values.

        world_to_camera
    """

    if extrinsics_direction == 'camera_to_world':
        R = np.linalg.inv(R)
        t = -R @ t

    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # Combine rotation and translation into a single transformation matrix
    extrinsic_matrix = np.hstack((R, t.reshape(-1, 1)))

    # Transform points from world coordinates to camera coordinates
    points_camera = (extrinsic_matrix @ points_homogeneous.T).T

    # Project points onto the image plane using the intrinsic matrix
    points_image_homogeneous = (K @ points_camera[:, :3].T).T

    # Normalize to get 2D image coordinates
    points_image = points_image_homogeneous[:, :2] / points_image_homogeneous[:, 2:]

    # The depth is the z-coordinate in the camera coordinate system
    depths = points_camera[:, 2]

    return points_image, depths


# Paths to the input files
files_path = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/'
intrinsics_file = files_path + 'intrinsics.txt'
extrinsics_file = files_path + 'DSC00350.JPG.txt'
points_file = files_path + 'xyz.txt'

# Read the data
K, width, height = read_intrinsics(intrinsics_file)
R, t = read_extrinsics(extrinsics_file)
points = read_points(points_file)

# Project the points and compute depths
projected_points, depths = project_points_to_image_plane(points, K, R, t)

# Save the transformed points to a .txt file
output_file_path = files_path + 'transformed_points.txt'
np.savetxt(output_file_path, np.hstack((points, projected_points, depths.reshape(-1, 1))), delimiter=' ',
           fmt='%.6f', header='x y z u v depth', comments='')

print(f"Transformed points and depths saved to {output_file_path}")



# Function to visualize depth
def visualize_depth(depths, projected_points, width, height, output_image_path):
    # Normalize depths to the range 0-255
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    normalized_depths = (depths - min_depth) / (max_depth - min_depth) * 255
    normalized_depths = normalized_depths.astype(np.uint8)

    # Create an empty image and fill it with maximum possible depth values (255)
    depth_image = np.full((height, width), 255, dtype=np.uint8)

    # Create an array to track the minimum depth at each pixel
    min_depths = np.full((height, width), np.inf)

    valid_points_count = 0
    for i, (u, v) in enumerate(projected_points):
        u, v = int(round(u)), int(round(v))
        if 0 <= u < width and 0 <= v < height:
            if normalized_depths[i] < min_depths[v, u]:
                min_depths[v, u] = normalized_depths[i]
                depth_image[v, u] = normalized_depths[i]
            valid_points_count += 1

    # get all inf value to be 0 to make it invalid
    for u in range(height):
        for v in range(width):
            if depth_image[v,u] > 254:
                depth_image[v, u] = 0

    # Save the depth image
    cv2.imwrite(output_image_path, depth_image)

    # Display the depth image
    plt.imshow(depth_image, cmap='gray')
    plt.title('Depth Image')
    plt.axis('off')
    plt.show()

    return valid_points_count
# Paths to the input files
files_path = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/depth_predicted/'
output_image_path = files_path + 'depth_image.jpg'

# Visualize the depth
valid_points_count = visualize_depth(depths, projected_points, width, height, output_image_path)

print(f"Depth image saved to {output_image_path}")
print(f"Number of valid points: {valid_points_count}")