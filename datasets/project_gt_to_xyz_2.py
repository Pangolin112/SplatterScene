import torch
import cv2
import numpy as np
import open3d as o3d

def get_user_input(prompt):
    return input(prompt)

# gt_img_idx = get_user_input('Enter the gt image index : ')
# pred_img_idx = get_user_input('Enter the predicted image index : ')
for gt_img_idx in range(350, 367):
    for pred_img_idx in range(350, 367):
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
            device = 'cuda'
            K = torch.tensor(K, device=device)
            K = K.to(device, dtype=torch.float32)

            return K, width, height

        def read_extrinsics(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                extrinsics = np.array([list(map(float, line.split())) for line in lines]).reshape((4, 4))
            R = extrinsics[:3, :3]
            t = extrinsics[:3, 3]

            device = 'cuda'
            R = torch.tensor(R, device=device)
            R = R.to(device, dtype=torch.float32)
            t = torch.tensor(t, device=device)
            t = t.to(device, dtype=torch.float32)

            return R, t

        def read_depth_image(file_path, device='cuda'):
            depth_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                raise ValueError(f"Could not read the image file {file_path}")
            depth_image = depth_image.astype(np.float32)
            depth_image = torch.tensor(depth_image, device=device)
            return depth_image

        def save_points_to_ply(points_world, file_path):
            points_world_np = points_world.cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world_np)
            o3d.io.write_point_cloud(file_path, pcd)

        def depth_image_to_world(depth_image, K, R, t, extrinsics_direction='world_to_camera', device='cuda'):
            depth_image = depth_image.to(device)
            if extrinsics_direction == 'camera_to_world':
                R = torch.inverse(R)
                t = -R @ t
            H, W = depth_image.shape
            u = torch.arange(0, W, device=device, dtype=torch.float32).view(1, -1).repeat(H, 1)
            v = torch.arange(0, H, device=device, dtype=torch.float32).view(-1, 1).repeat(1, W)
            u = u.view(-1)
            v = v.view(-1)
            depths = depth_image.view(-1)
            uv1 = torch.stack([u, v, torch.ones_like(u)], dim=1)
            K_inv = torch.inverse(K)
            normalized_coords = (K_inv @ uv1.T).T
            points_camera = normalized_coords * depths.view(-1, 1)
            ones = torch.ones((points_camera.shape[0], 1), device=device)
            points_camera_homogeneous = torch.cat((points_camera, ones), dim=1)
            extrinsic_matrix = torch.cat((R, t.reshape(-1, 1)), dim=1)
            if extrinsics_direction == 'camera_to_world':
                extrinsic_matrix_inv = torch.inverse(extrinsic_matrix)
            else:
                extrinsic_matrix_inv = torch.cat((torch.inverse(R), (-torch.inverse(R) @ t).reshape(-1, 1)), dim=1)
            points_world_homogeneous = (extrinsic_matrix_inv @ points_camera_homogeneous.T).T
            points_world = points_world_homogeneous[:, :3]
            return points_world.view(H, W, 3)

        def project_points_to_image_plane(points, K, R, t, extrinsics_direction='world_to_camera', device='cuda'):
            points = points.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)
            if extrinsics_direction == 'camera_to_world':
                R = torch.inverse(R)
                t = -R @ t
            ones = torch.ones((points.shape[0], 1), device=device)
            points_homogeneous = torch.cat((points, ones), dim=1)
            extrinsic_matrix = torch.cat((R, t.reshape(-1, 1)), dim=1)
            points_camera = (extrinsic_matrix @ points_homogeneous.T).T
            points_image_homogeneous = (K @ points_camera[:, :3].T).T
            points_image = points_image_homogeneous[:, :2] / points_image_homogeneous[:, 2:]
            depths = points_camera[:, 2]
            return points_image, depths

        def visualize_depth(depths, projected_points,pred_img_idx, width=128, height=128, device='cuda'):
            depths = depths.contiguous().to(device)
            projected_points = projected_points.contiguous().to(device)
            min_depth = torch.min(depths)
            max_depth = torch.max(depths)
            normalized_depths = (depths - min_depth) / (max_depth - min_depth) * 255
            normalized_depths = normalized_depths.to(torch.uint8)
            depth_image = torch.full((height, width), float('inf'), dtype=torch.float32, device=device)
            projected_points = projected_points.round().long()
            valid_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < width) & \
                         (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height)
            projected_points = projected_points[valid_mask]
            normalized_depths = normalized_depths[valid_mask]
            v = projected_points[:, 0]
            u = projected_points[:, 1]
            depth_image[u, v] = torch.min(depth_image[u, v], normalized_depths)
            depth_image[depth_image == float('inf')] = 0
            depth_image_np = depth_image.cpu().numpy()
            file_path = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/depth_from_gt/'
            output_image_path = file_path + 'DSC00' + str(pred_img_idx) + '.jpg'
            cv2.imwrite(output_image_path, depth_image_np)
            return depth_image

        # File paths and data reading
        files_path = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/'
        depth_path = files_path + 'depth/DSC00'+str(gt_img_idx)+'.jpg'
        intrinsics_file = files_path + 'intrinsics.txt'
        extrinsics_file = files_path + 'pose_colmap/DSC00'+str(gt_img_idx)+'.JPG.txt'
        extrinsics_novel_file = files_path + 'pose_colmap/DSC00'+str(pred_img_idx)+'.JPG.txt'
        ply_file_path = files_path + 'xyz_from_gt/DSC00'+str(gt_img_idx)+'.ply'


        # Correct file paths for extrinsics
        R, t = read_extrinsics(extrinsics_file)
        R_novel, t_novel = read_extrinsics(extrinsics_novel_file)

        K, width, height = read_intrinsics(intrinsics_file)
        camera_to_world = 'world_to_camera'
        depth_image = read_depth_image(depth_path)
        points_world = depth_image_to_world(depth_image, K, R, t, camera_to_world)

        #txt_file_path = '/home/qianru/Projects/TUM/TUM_2/ADL4CV/Data/ScanNetpp/data_1/0cf2e9402d/xyz_predicted/predicted_xyz550.txt'
        #points_world = torch.tensor(np.loadtxt(txt_file_path)).to('cuda', dtype=torch.float32)



        # Flatten the points_world tensor to (HxW)x3
        points_world_flat = points_world.view(-1, 3)
        save_points_to_ply(points_world_flat, ply_file_path)

        points_image, depths = project_points_to_image_plane(points_world_flat, K, R_novel, t_novel, camera_to_world)
        vis_image = visualize_depth(depths, points_image,pred_img_idx)
