import glob
import hydra
import os
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader

from lightning.fabric import Fabric

from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, l2_loss
import lpips as lpips_lib

from eval import evaluate_dataset
from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from datasets.dataset_factory import get_dataset

import open3d as o3d

import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import math
import time
import datasets.base_path as bp

#########################for gradients, run much slower!!!!!!!!!!!!!!!!!!######################
#torch.autograd.set_detect_anomaly(True)
#########################for gradients, run much slower!!!!!!!!!!!!!!!!!!######################

# +experiment=lpips_100k.yaml

current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '/'
############ for depth #################################
save_iterations = bp.save_iterations

output_base_path = bp.base_path + bp.scene_name + '/'

def project_points_to_image_plane(points, K, R, t, iteration, extrinsics_direction='world_to_camera', device='cuda'):
    points = points.to(device)
    points = points.reshape(-1, 3)

    file_path = output_base_path + 'xyz_predicted/'
    file_path = file_path + str(current_time)
    if iteration == 2:
        os.makedirs(file_path, exist_ok=True)

    if iteration % save_iterations == 0:
        points_clone = points.clone()
        points_np = points_clone.detach().cpu().numpy()
        txt_path = file_path + 'predicted_xyz' + str(iteration) + '.txt'
        with open(txt_path, 'w') as f:
            for point in points_np:
                f.write(f"{point[0].item()} {point[1].item()} {point[2].item()}\n")

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


def visualize_depth(depths, projected_points, iteration, width=128, height=128, device='cuda'):
    depths = depths.contiguous().to(device)
    projected_points = projected_points.contiguous().to(device)

    depth_image = torch.full((height, width), float('inf'), dtype=torch.float32, device=device)
    projected_points = projected_points.round().long()
    valid_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < width) & \
                 (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height)
    projected_points = projected_points[valid_mask]
    depths = depths[valid_mask]

    v = projected_points[:, 0]
    u = projected_points[:, 1]
    depth_image[u, v] = torch.min(depth_image[u, v], depths)
    depth_image[depth_image == float('inf')] = 0

    file_path = output_base_path + 'depth_predicted/'
    file_path = file_path + str(current_time)
    if iteration == 2:
        os.makedirs(file_path, exist_ok=True)

    if iteration % save_iterations == 0:
        depth_image_np = depth_image.cpu().detach().numpy()
        output_image_path = file_path + 'depth_image' + str(iteration) + '.jpg'
        cv2.imwrite(output_image_path, depth_image_np)

    # Create a mask for the predicted depth image where the pixel value is 0 (black pixels)
    mask = (depth_image > 0).float()

    return depth_image, mask


def depth_image_to_world(depth_image, K, R, t, iteration, extrinsics_direction='world_to_camera', device='cuda'):
    depth_image = depth_image.to(device)
    if extrinsics_direction == 'camera_to_world':
        R = torch.inverse(R)
        t = -R @ t
    _, H, W = depth_image.shape
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
    extrinsic_matrix = torch.eye(4, device=device)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = t.view(-1)
    extrinsic_matrix_inv = torch.inverse(extrinsic_matrix)
    points_world_homogeneous = (extrinsic_matrix_inv @ points_camera_homogeneous.T).T
    points_world = points_world_homogeneous[:, :3]

    file_path = output_base_path + 'ply_gt/'
    file_path = file_path + str(current_time)
    if iteration == 2:
        os.makedirs(file_path, exist_ok=True)

    if iteration % save_iterations == 0:
        points_np = points_world.detach().cpu().numpy()

        # Create an Open3D point cloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_np)

        ply_path = file_path + str(iteration) + ".ply"

        # Save the point cloud to a .ply file
        o3d.io.write_point_cloud(ply_path, point_cloud)

    return points_world.view(H, W, 3)


def rotation_matrix_to_euler_angles(R):
    assert R.shape == (3, 3), "Input rotation matrix must be 3x3"

    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z])


def align_point_clouds(source_points, target_points, iteration):
    source_points = source_points.reshape(-1, 3)
    target_points = target_points.reshape(-1, 3)


    # Normalize the point clouds to unit size
    source_mean = torch.mean(source_points, dim=0, keepdim=True)
    target_mean = torch.mean(target_points, dim=0, keepdim=True)

    source_points_normalized = source_points - source_mean
    target_points_normalized = target_points - target_mean

    source_scale = torch.norm(source_points_normalized)
    target_scale = torch.norm(target_points_normalized)
    ###########################Do not use inplace operation#########################################
    source_points_normalized = source_points_normalized / source_scale
    target_points_normalized = target_points_normalized / target_scale
    ###########################Do not use inplace operation#########################################

    # Apply Procrustes Analysis for scaling and rotation
    M = torch.mm(source_points_normalized.t(), target_points_normalized)
    U, S, Vt = torch.svd(M)
    R = torch.mm(U, Vt.t())

    # with torch.no_grad():
    source_points_aligned = torch.mm(source_points_normalized, R)

    # Scale back and translate
    source_points_aligned_final = source_points_aligned * target_scale + target_mean

    file_path = output_base_path + 'aligned_predicted_ply/'
    file_path = file_path + str(current_time)

    transformation_path = file_path + '/tranformation/'

    if iteration == 2:
        os.makedirs(file_path, exist_ok=True)
        os.makedirs(transformation_path, exist_ok=True)

    if iteration % save_iterations == 0:
        points_np = source_points_aligned_final.detach().cpu().numpy()

        # Create an Open3D point cloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_np)

        ply_path = file_path + str(iteration) + ".ply"

        # Save the point cloud to a .ply file
        o3d.io.write_point_cloud(ply_path, point_cloud)

    translation_matrix = target_mean - torch.mm(source_mean, R.t()) * target_scale
    euler_angles = rotation_matrix_to_euler_angles(R)
    print("Rotation Matrix:", R)
    print("Translation Matrix:", translation_matrix)
    print("Euler Angles (rad):", euler_angles)
    print("Euler Angles (deg):", euler_angles * 180 / torch.pi)
    print(f"Iteration {iteration}: Source Scale = {source_scale.item()}, Target Scale = {target_scale.item()}", " scale times: ", target_scale.item() / source_scale.item())
    data = f"""
    Iteration {iteration}: Source Scale = {source_scale.item()}, Target Scale = {target_scale.item()}, scale times: {target_scale.item() / source_scale.item()}
    Rotation Matrix: {R}
    Translation Matrix: {translation_matrix}
    Euler Angles (rad): {euler_angles}
    Euler Angles (deg): {euler_angles * 180 / torch.pi}
    """
    transformation_txt_path = transformation_path +str(iteration) + ".txt"
    with open(transformation_txt_path, 'a') as file:
        file.write(data)

    return source_points_aligned_final

def manual_align_point_clouds(source_points, target_points, rotation_degrees, rescale_number, translation, iteration):

    device = source_points.device

    source_points = source_points.reshape(-1, 3)
    target_points = target_points.reshape(-1, 3)

    # Convert rotation degrees to radians
    rotation_radians = [math.radians(deg) for deg in rotation_degrees]

    # Create rotation matrices for each axis
    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(rotation_radians[0]), -math.sin(rotation_radians[0])],
        [0, math.sin(rotation_radians[0]), math.cos(rotation_radians[0])]
    ], dtype=torch.float32, device=device)

    Ry = torch.tensor([
        [math.cos(rotation_radians[1]), 0, math.sin(rotation_radians[1])],
        [0, 1, 0],
        [-math.sin(rotation_radians[1]), 0, math.cos(rotation_radians[1])]
    ], dtype=torch.float32, device=device)

    Rz = torch.tensor([
        [math.cos(rotation_radians[2]), -math.sin(rotation_radians[2]), 0],
        [math.sin(rotation_radians[2]), math.cos(rotation_radians[2]), 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    # Combine rotations
    R = torch.mm(Rz, torch.mm(Ry, Rx))

    # Apply rotation
    source_points_rotated = torch.mm(source_points, R.t())

    # Apply scaling
    source_points_scaled = source_points_rotated * rescale_number

    # Compute centroids of source and target points
    source_centroid = torch.mean(source_points_scaled, dim=0)
    target_centroid = torch.mean(target_points, dim=0)

    # Adjust translation vector to align centroids
    translation_vector = target_centroid - source_centroid
    # translation_vector = torch.tensor(translation, dtype=torch.float32, device=device)

    # translation_vector[0] = translation_vector[0] + torch.abs(translation_vector[0] * 0.5)
    # translation_vector[1] = translation_vector[1] - torch.abs(translation_vector[1] * 0.3)

    # Apply translation
    source_points_transformed = source_points_scaled + translation_vector
    # source_points_transformed = source_points_scaled  # no translation
    # print(f'{iteration} translation: {translation_vector}')

    file_path = output_base_path + 'aligned_predicted_ply/'
    file_path = file_path + str(current_time)

    if iteration == 2:
        os.makedirs(file_path, exist_ok=True)

    if iteration % save_iterations == 0:
        points_np = source_points_transformed.detach().cpu().numpy()

        # Create an Open3D point cloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_np)

        ply_path = file_path + str(iteration) + ".ply"

        # Save the point cloud to a .ply file
        o3d.io.write_point_cloud(ply_path, point_cloud)

    return source_points_transformed
############ for depth #################################


@hydra.main(version_base=None, config_path='configs', config_name="default_config")
def main(cfg: DictConfig):

    torch.set_float32_matmul_precision('high')
    if cfg.general.mixed_precision:
        fabric = Fabric(accelerator="cuda", devices=cfg.general.num_devices, strategy="ddp",
                        precision="16-mixed")
    else:
        fabric = Fabric(accelerator="cuda", devices=cfg.general.num_devices, strategy="ddp")
    fabric.launch()

    if fabric.is_global_zero:
        vis_dir = os.getcwd()

        dict_cfg = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

        if os.path.isdir(os.path.join(vis_dir, "wandb")):
            run_name_path = glob.glob(os.path.join(vis_dir, "wandb", "latest-run", "run-*"))[0]
            print("Got run name path {}".format(run_name_path))
            run_id = os.path.basename(run_name_path).split("run-")[1].split(".wandb")[0]
            print("Resuming run with id {}".format(run_id))
            wandb_run = wandb.init(project=cfg.wandb.project, resume=True,
                            id = run_id, config=dict_cfg)

        else:
            wandb_run = wandb.init(project=cfg.wandb.project, reinit=True,
                            config=dict_cfg)

    first_iter = 0
    device = safe_state(cfg)

    gaussian_predictor = GaussianSplatPredictor(cfg)
    gaussian_predictor = gaussian_predictor.to(memory_format=torch.channels_last)

    l = []
    if cfg.model.network_with_offset:
        l.append({'params': gaussian_predictor.network_with_offset.parameters(),
         'lr': cfg.opt.base_lr})
    if cfg.model.network_without_offset:
        l.append({'params': gaussian_predictor.network_wo_offset.parameters(),
         'lr': cfg.opt.base_lr})
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15,
                                 betas=cfg.opt.betas)

    # Resuming training
    if fabric.is_global_zero:
        if os.path.isfile(os.path.join(vis_dir, "model_latest.pth")):
            print('Loading an existing model from ', os.path.join(vis_dir, "model_latest.pth"))
            checkpoint = torch.load(os.path.join(vis_dir, "model_latest.pth"),
                                    map_location=device)
            try:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"],
                                                strict=False)
                print("Warning, model mismatch - was this expected?")
            first_iter = checkpoint["iteration"]
            best_PSNR = checkpoint["best_PSNR"]
            print('Loaded model')
        # Resuming from checkpoint
        elif cfg.opt.pretrained_ckpt is not None:
            pretrained_ckpt_dir = os.path.join(cfg.opt.pretrained_ckpt, "model_latest.pth")
            checkpoint = torch.load(pretrained_ckpt_dir,
                                    map_location=device)
            try:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"],
                                                strict=False)
            best_PSNR = checkpoint["best_PSNR"]
            print('Loaded model from a pretrained checkpoint')
        else:
            best_PSNR = 0.0

    if cfg.opt.ema.use and fabric.is_global_zero:
        ema = EMA(gaussian_predictor,
                  beta=cfg.opt.ema.beta,
                  update_every=cfg.opt.ema.update_every,
                  update_after_step=cfg.opt.ema.update_after_step)
        ema = fabric.to_device(ema)

    if cfg.opt.loss == "l2":
        loss_fn = l2_loss
    elif cfg.opt.loss == "l1":
        loss_fn = l1_loss

    if cfg.opt.lambda_lpips != 0:
        lpips_fn = fabric.to_device(lpips_lib.LPIPS(net='vgg'))
    lambda_lpips = cfg.opt.lambda_lpips
    lambda_l12 = 1.0 - lambda_lpips

    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32)
    background = fabric.to_device(background)

    if cfg.data.category in ["nmr", "objaverse"]:
        num_workers = 12
        persistent_workers = True
    else:
        num_workers = 0
        persistent_workers = False

    dataset = get_dataset(cfg, "train")
    dataloader = DataLoader(dataset,
                            batch_size=cfg.opt.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            persistent_workers=persistent_workers)

    val_dataset = get_dataset(cfg, "val")
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                persistent_workers=True,
                                pin_memory=True)

    test_dataset = get_dataset(cfg, "vis")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=True)

    # distribute model and training dataset
    gaussian_predictor, optimizer = fabric.setup(
        gaussian_predictor, optimizer
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    gaussian_predictor.train()

    print("Beginning training")
    first_iter += 1
    iteration = first_iter

    for num_epoch in range((cfg.opt.iterations + 1 - first_iter) // len(dataloader) + 1):
        dataloader.sampler.set_epoch(num_epoch)
        for data in dataloader:
            iteration += 1

            print("starting iteration {} on process {}".format(iteration, fabric.global_rank))

            # =============== Prepare input ================
            rot_transform_quats = data["source_cv2wT_quat"][:, :cfg.data.input_images]

            if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                focals_pixels_pred = data["focals_pixels"][:, :cfg.data.input_images, ...]
                input_images = torch.cat([data["gt_images"][:, :cfg.data.input_images, ...],
                                data["origin_distances"][:, :cfg.data.input_images, ...]],
                                dim=2)
            else:
                focals_pixels_pred = None
                input_images = data["gt_images"][:, :cfg.data.input_images, ...]

            gaussian_splats = gaussian_predictor(input_images,
                                                data["view_to_world_transforms"][:, :cfg.data.input_images, ...],
                                                rot_transform_quats,
                                                focals_pixels_pred)


            if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                # regularize very big gaussians
                if len(torch.where(gaussian_splats["scaling"] > 20)[0]) > 0:
                    big_gaussian_reg_loss = torch.mean(
                        gaussian_splats["scaling"][torch.where(gaussian_splats["scaling"] > 20)] * 0.1)
                    print('Regularising {} big Gaussians on iteration {}'.format(
                        len(torch.where(gaussian_splats["scaling"] > 20)[0]), iteration))
                else:
                    big_gaussian_reg_loss = 0.0
                # regularize very small Gaussians
                if len(torch.where(gaussian_splats["scaling"] < 1e-5)[0]) > 0:
                    small_gaussian_reg_loss = torch.mean(
                        -torch.log(gaussian_splats["scaling"][torch.where(gaussian_splats["scaling"] < 1e-5)]) * 0.1)
                    print('Regularising {} small Gaussians on iteration {}'.format(
                        len(torch.where(gaussian_splats["scaling"] < 1e-5)[0]), iteration))
                else:
                    small_gaussian_reg_loss = 0.0
            # Render
            l12_loss_sum = 0.0
            lpips_loss_sum = 0.0
            depth_loss_sum = 0.0
            rendered_images = []
            gt_images = []
            predicted_depth_images = []
            gt_depth_images = []
            for b_idx in range(data["gt_images"].shape[0]):
                # image at index 0 is training, remaining images are targets
                # Rendering is done sequentially because gaussian rasterization code
                # does not support batching
                gaussian_splat_batch = {k: v[b_idx].contiguous() for k, v in gaussian_splats.items()}

                ############ for depth #################################
                points = gaussian_splat_batch["xyz"]
                K_input = data["Ks"][b_idx, 0]
                R_input = data["colmap_depth_Rs"][b_idx, 0]
                T_input = data["colmap_depth_Ts"][b_idx, 0]
                gt_depth_input_image = data["gt_depths"][b_idx, 0] * 255.0
                gt_points = depth_image_to_world(gt_depth_input_image, K_input, R_input, T_input, iteration)
                # aligned_points = align_point_clouds(points, gt_points, iteration) # auto align
                # aligned_points = manual_align_point_clouds(points, gt_points, [-90, 0, -90], 15, [-80, 0, 0], iteration)  # manually align
                aligned_points = manual_align_point_clouds(points, gt_points, [-90, 0, -90], 15, [0, 0, 0], iteration)
                ############ for depth #################################

                for r_idx in range(cfg.data.input_images, data["gt_images"].shape[1]):
                    if "focals_pixels" in data.keys():
                        focals_pixels_render = data["focals_pixels"][b_idx, r_idx].cpu()
                    else:
                        focals_pixels_render = None

                    ############ for depth #################################
                    K = data["Ks"][b_idx, r_idx]
                    R = data["colmap_depth_Rs"][b_idx, r_idx]
                    T = data["colmap_depth_Ts"][b_idx, r_idx]
                    projected_points, predicted_depths = project_points_to_image_plane(aligned_points, K, R, T, iteration)
                    predicted_depth_image, mask_predicted = visualize_depth(predicted_depths, projected_points, iteration)

                    # directly use the gt depth of each view
                    gt_depth_image = data["gt_depths"][b_idx, r_idx] * 255.0
                    mask_gt = (gt_depth_image > 5.0).float()

                    # mask gt depth
                    masked_gt_depth = gt_depth_image * mask_predicted
                    gt_depth_images.append(masked_gt_depth)

                    # mask predicted depth
                    masked_predicted_depth = predicted_depth_image * mask_gt
                    predicted_depth_images.append(masked_predicted_depth)
                    ############ for depth #################################

                    image = render_predicted(gaussian_splat_batch,
                                        data["world_view_transforms"][b_idx, r_idx],
                                        data["full_proj_transforms"][b_idx, r_idx],
                                        data["camera_centers"][b_idx, r_idx],
                                        background,
                                        cfg,
                                        focals_pixels=focals_pixels_render)["render"]

                    # ================== for rgb saving ==================
                    file_path = output_base_path + 'rgb_predicted/'
                    file_path = file_path + str(current_time)
                    if iteration == 2:
                        os.makedirs(file_path, exist_ok=True)

                    if iteration % save_iterations == 0:
                        # Convert tensor to NumPy array and scale to [0, 255]
                        rgb_image_np = image.cpu().detach().numpy() * 255.0
                        # Transpose to [height, width, channels]
                        rgb_image_np = rgb_image_np.transpose(1, 2, 0)  # From [3, 128, 128] to [128, 128, 3]
                        # Convert RGB to BGR
                        bgr_image_np = cv2.cvtColor(rgb_image_np.astype('uint8'), cv2.COLOR_RGB2BGR)
                        # Define the output image path
                        output_image_path = file_path + 'rgb_image' + str(iteration) + '.jpg'
                        # Save the image using OpenCV
                        cv2.imwrite(output_image_path, bgr_image_np)
                    # ================== for rgb saving ==================

                    # Put in a list for a later loss computation
                    rendered_images.append(image)
                    gt_image = data["gt_images"][b_idx, r_idx]
                    gt_images.append(gt_image)

            rendered_images = torch.stack(rendered_images, dim=0)
            gt_images = torch.stack(gt_images, dim=0)

            gt_depth_images = torch.stack(gt_depth_images, dim=0)
            predicted_depth_images = torch.stack(predicted_depth_images, dim=0)

            # Loss computation
            # rgb
            l12_loss_sum = loss_fn(rendered_images, gt_images)
            # depth
            depth_loss_sum = loss_fn(predicted_depth_images, gt_depth_images)
            # mask_predicted
            mask_reg_loss = torch.mean(1 - mask_predicted)

            # lambda coefficients
            # rgb
            # lambda_l12 = 0.0
            # lpips
            if cfg.opt.lambda_lpips != 0:
                lpips_loss_sum = torch.mean(
                    lpips_fn(rendered_images * 2 - 1, gt_images * 2 - 1),
                    )
            # depth
            lambda_depth = 0.0001
            # lambda_depth = 0.000
            # mask
            lambda_mask = 0.01
            # lambda_mask = 0.0

            total_loss = l12_loss_sum * lambda_l12 + lpips_loss_sum * lambda_lpips + depth_loss_sum * lambda_depth + mask_reg_loss * lambda_mask

            print('training loss is: ', total_loss.item() * 100)
            print('l12 loss is     : ', l12_loss_sum.item() * 100)
            print('depth loss is   : ', depth_loss_sum.item() * lambda_depth * 100)
            print('mask loss is    : ', mask_reg_loss.item() * lambda_mask * 100)

            if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                total_loss = total_loss + big_gaussian_reg_loss + small_gaussian_reg_loss

            assert not total_loss.isnan(), "Found NaN loss!"
            print("finished forward {} on process {}".format(iteration, fabric.global_rank))
            fabric.backward(total_loss)

            # ============ Optimization ===============
            optimizer.step()
            optimizer.zero_grad()
            print("finished opt {} on process {}".format(iteration, fabric.global_rank))

            if cfg.opt.ema.use and fabric.is_global_zero:
                ema.update()

            print("finished iteration {} on process {}".format(iteration, fabric.global_rank))

            gaussian_predictor.eval()

            # ========= Logging =============
            with torch.no_grad():
                if iteration % cfg.logging.loss_log == 0 and fabric.is_global_zero:
                    wandb.log({"training_loss": np.log10(total_loss.item() + 1e-8)}, step=iteration)
                    wandb.log({"training_l12_loss": np.log10(l12_loss_sum.item() + 1e-8)}, step=iteration)
                    wandb.log({"training_depth_loss": np.log10(depth_loss_sum.item() + 1e-8)}, step=iteration)
                    wandb.log({"training_mask_loss": np.log10(mask_reg_loss.item() + 1e-8)}, step=iteration)
                    if cfg.opt.lambda_lpips != 0:
                        wandb.log({"training_lpips_loss": np.log10(lpips_loss_sum.item() + 1e-8)}, step=iteration)
                    if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                        if type(big_gaussian_reg_loss) == float:
                            brl_for_log = big_gaussian_reg_loss
                        else:
                            brl_for_log = big_gaussian_reg_loss.item()
                        if type(small_gaussian_reg_loss) == float:
                            srl_for_log = small_gaussian_reg_loss
                        else:
                            srl_for_log = small_gaussian_reg_loss.item()
                        wandb.log({"reg_loss_big": np.log10(brl_for_log + 1e-8)}, step=iteration)
                        wandb.log({"reg_loss_small": np.log10(srl_for_log + 1e-8)}, step=iteration)

                if (iteration % cfg.logging.render_log == 0 or iteration == 1) and fabric.is_global_zero:
                    wandb.log({"render": wandb.Image(image.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy())}, step=iteration)
                    wandb.log({"gt": wandb.Image(gt_image.permute(1, 2, 0).detach().cpu().numpy())}, step=iteration)
                if (iteration % cfg.logging.loop_log == 0 or iteration == 1) and fabric.is_global_zero:
                    # torch.cuda.empty_cache()
                    try:
                        vis_data = next(test_iterator)
                    except UnboundLocalError:
                        test_iterator = iter(test_dataloader)
                        vis_data = next(test_iterator)
                    except StopIteration or UnboundLocalError:
                        test_iterator = iter(test_dataloader)
                        vis_data = next(test_iterator)

                    vis_data = {k: fabric.to_device(v) for k, v in vis_data.items()}

                    rot_transform_quats = vis_data["source_cv2wT_quat"][:, :cfg.data.input_images]

                    if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                        focals_pixels_pred = vis_data["focals_pixels"][:, :cfg.data.input_images, ...]
                        input_images = torch.cat([vis_data["gt_images"][:, :cfg.data.input_images, ...],
                                                vis_data["origin_distances"][:, :cfg.data.input_images, ...]],
                                                dim=2)
                    else:
                        focals_pixels_pred = None
                        input_images = vis_data["gt_images"][:, :cfg.data.input_images, ...]

                    gaussian_splats_vis = gaussian_predictor(input_images,
                                                        vis_data["view_to_world_transforms"][:, :cfg.data.input_images, ...],
                                                        rot_transform_quats,
                                                        focals_pixels_pred)

                    test_loop = []
                    test_loop_gt = []
                    for r_idx in range(vis_data["gt_images"].shape[1]):
                        # We don't change the input or output of the network, just the rendering cameras
                        if "focals_pixels" in vis_data.keys():
                            focals_pixels_render = vis_data["focals_pixels"][0, r_idx]
                        else:
                            focals_pixels_render = None
                        test_image = render_predicted({k: v[0].contiguous() for k, v in gaussian_splats_vis.items()},
                                            vis_data["world_view_transforms"][0, r_idx],
                                            vis_data["full_proj_transforms"][0, r_idx],
                                            vis_data["camera_centers"][0, r_idx],
                                            background,
                                            cfg,
                                            focals_pixels=focals_pixels_render)["render"]
                        test_loop_gt.append((np.clip(vis_data["gt_images"][0, r_idx].detach().cpu().numpy(), 0, 1)*255).astype(np.uint8))
                        test_loop.append((np.clip(test_image.detach().cpu().numpy(), 0, 1)*255).astype(np.uint8))

                    wandb.log({"rot": wandb.Video(np.asarray(test_loop), fps=20, format="mp4")},
                        step=iteration)
                    wandb.log({"rot_gt": wandb.Video(np.asarray(test_loop_gt), fps=20, format="mp4")},
                        step=iteration)
            torch.cuda.empty_cache()


            fnames_to_save = []
            # Find out which models to save
            if (iteration + 1) % cfg.logging.ckpt_iterations == 0 and fabric.is_global_zero:
                fnames_to_save.append("model_latest.pth")
            if (iteration + 1) % cfg.logging.val_log == 0 and fabric.is_global_zero:
                torch.cuda.empty_cache()
                print("\n[ITER {}] Validating".format(iteration + 1))
                if cfg.opt.ema.use:
                    scores = evaluate_dataset(
                        ema,
                        val_dataloader,
                        device=device,
                        model_cfg=cfg)
                else:
                    scores = evaluate_dataset(
                        gaussian_predictor,
                        val_dataloader,
                        device=device,
                        model_cfg=cfg)
                wandb.log(scores, step=iteration+1)
                # save models - if the newest psnr is better than the best one,
                # overwrite best_model. Always overwrite the latest model.
                if scores["PSNR_novel"] > best_PSNR:
                    fnames_to_save.append("model_best.pth")
                    best_PSNR = scores["PSNR_novel"]
                    print("\n[ITER {}] Saving new best checkpoint PSNR:{:.2f}".format(
                        iteration + 1, best_PSNR))
                torch.cuda.empty_cache()

            # ============ Model saving =================
            for fname_to_save in fnames_to_save:
                ckpt_save_dict = {
                                "iteration": iteration,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": total_loss.item(),
                                "best_PSNR": best_PSNR
                                }
                if cfg.opt.ema.use:
                    ckpt_save_dict["model_state_dict"] = ema.ema_model.state_dict()
                else:
                    ckpt_save_dict["model_state_dict"] = gaussian_predictor.state_dict()
                torch.save(ckpt_save_dict, os.path.join(vis_dir, fname_to_save))

            gaussian_predictor.train()

    wandb_run.finish()

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()