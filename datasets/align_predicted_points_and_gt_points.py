import open3d as o3d
import numpy as np
import torch

'''
# Load point clouds
pcd1 = o3d.io.read_point_cloud("/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/xyz_from_gt/DSC00358.ply")
pcd2 = o3d.io.read_point_cloud("/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/ply_predicted/1100.ply")

# Function to align two point clouds using scale, rotation, and translation
def align_point_clouds(source, target):
    # Normalize the point clouds to unit size
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)

    source_points -= source_mean
    target_points -= target_mean

    source_scale = np.linalg.norm(source_points)
    target_scale = np.linalg.norm(target_points)

    source_points /= source_scale
    target_points /= target_scale

    # Apply Procrustes Analysis for scaling and rotation
    M = np.dot(source_points.T, target_points)
    U, S, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)

    source_points_aligned = np.dot(source_points, R)

    # Scale back and translate
    source_points_aligned = source_points_aligned * target_scale + target_mean

    # Create a new point cloud with the aligned points
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(source_points_aligned)

    return aligned_pcd

# Align the point clouds
aligned_pcd = align_point_clouds(pcd2, pcd1)

# Save the aligned point cloud
o3d.io.write_point_cloud("/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/aligned_predicted_ply/aligned_1100.ply", aligned_pcd)

# Visualize the point clouds
o3d.visualization.draw_geometries([pcd1, aligned_pcd], window_name="Alignment Result", width=800, height=600)
'''


def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return torch.tensor(points, dtype=torch.float32)


pcd1_file_path = "/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/xyz_from_gt/DSC00358.ply"
pcd2_file_path = "/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/ply_predicted/1100.ply"

pcd1_points = load_point_cloud(pcd1_file_path)
pcd2_points = load_point_cloud(pcd2_file_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pcd1_points = pcd1_points.to(device)
pcd2_points = pcd2_points.to(device)

def align_point_clouds_torch(source_points, target_points):
    # Normalize the point clouds to unit size
    source_mean = torch.mean(source_points, dim=0)
    target_mean = torch.mean(target_points, dim=0)

    source_points_normalized = source_points - source_mean
    target_points_normalized = target_points - target_mean

    source_scale = torch.norm(source_points_normalized)
    target_scale = torch.norm(target_points_normalized)

    source_points_normalized /= source_scale
    target_points_normalized /= target_scale

    # Apply Procrustes Analysis for scaling and rotation
    M = torch.mm(source_points_normalized.t(), target_points_normalized)
    U, S, Vt = torch.svd(M)
    R = torch.mm(U, Vt.t())

    source_points_aligned = torch.mm(source_points_normalized, R)

    # Scale back and translate
    source_points_aligned = source_points_aligned * target_scale + target_mean

    return source_points_aligned


aligned_pcd2_points = align_point_clouds_torch(pcd2_points, pcd1_points)

# Convert back to Open3D point cloud for saving and visualization
aligned_pcd2 = o3d.geometry.PointCloud()
aligned_pcd2.points = o3d.utility.Vector3dVector(aligned_pcd2_points.cpu().numpy())

# Save the aligned point cloud
o3d.io.write_point_cloud("/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/aligned_predicted_ply/torch_aligned_1100.ply", aligned_pcd2)

# Visualize the point clouds
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(pcd1_points.cpu().numpy())

o3d.visualization.draw_geometries([pcd1, aligned_pcd2], window_name="Alignment Result", width=800, height=600)
