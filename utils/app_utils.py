from PIL import Image
from typing import Any
import rembg
import numpy as np
from torchvision import transforms
from plyfile import PlyData, PlyElement
import os
import torch
from .camera_utils import get_loop_cameras
from .graphics_utils import getProjectionMatrix
from .general_utils import matrix_to_quaternion, quaternion_raw_multiply
import math

def remove_background(image, rembg_session):
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    if do_remove:
        image = rembg.remove(image, session=rembg_session)
    return image

def set_white_background(image):
    image = np.array(image).astype(np.float32) / 255.0
    mask = image[:, :, 3:4]
    image = image[:, :, :3] * mask + (1 - mask)
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image

def resize_foreground(image, ratio):
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    # modify so that cropping doesn't change the world center
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )

    # crop the foreground
    fg = image[y1: y2, 
               x1: x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((255, 255), (255, 255), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((255, 255), (255, 255), (0, 0)),
    )

    new_image = Image.fromarray(new_image)

    return new_image

def resize_to_128(img):
    img = transforms.functional.resize(img, 128,
        interpolation=transforms.InterpolationMode.LANCZOS)
    return img

def to_tensor(img):
    img = torch.tensor(img).permute(2, 0, 1) / 255.0
    return img

def get_source_camera_v2w_rmo_and_quats(num_imgs_in_loop=200):
    source_camera = get_loop_cameras(num_imgs_in_loop=num_imgs_in_loop)[0]
    source_camera = torch.from_numpy(source_camera).transpose(0, 1).unsqueeze(0)
    
    qs = []
    for c_idx in range(source_camera.shape[0]):
        qs.append(matrix_to_quaternion(source_camera[c_idx, :3, :3].transpose(0, 1)))
    
    return source_camera.unsqueeze(0), torch.stack(qs, dim=0).unsqueeze(0)

def get_target_cameras(num_imgs_in_loop=200):
    """
    Returns camera parameters for rendering a loop around the object:
      world_to_view_transforms, 
      full_proj_transforms,
      camera_centers
    """

    projection_matrix = getProjectionMatrix(
        znear=0.8, zfar=3.2,
        fovX=49.134342641202636 * 2 * np.pi / 360, 
        fovY=49.134342641202636 * 2 * np.pi / 360).transpose(0,1)

    target_cameras = get_loop_cameras(num_imgs_in_loop=num_imgs_in_loop,
                                      max_elevation=np.pi/4,
                                      elevation_freq=1.5)
    world_view_transforms = []
    view_world_transforms = []
    camera_centers = []
    
    for loop_camera_c2w_cmo in target_cameras:
        view_world_transform = torch.from_numpy(loop_camera_c2w_cmo).transpose(0, 1)
        world_view_transform = torch.from_numpy(loop_camera_c2w_cmo).inverse().transpose(0, 1)
        camera_center = view_world_transform[3, :3].clone()
    
        world_view_transforms.append(world_view_transform)
        view_world_transforms.append(view_world_transform)
        camera_centers.append(camera_center)

    world_view_transforms = torch.stack(world_view_transforms)
    view_world_transforms = torch.stack(view_world_transforms)
    camera_centers = torch.stack(camera_centers)

    full_proj_transforms = world_view_transforms.bmm(projection_matrix.unsqueeze(0).expand(
        world_view_transforms.shape[0], 4, 4))

    return world_view_transforms, full_proj_transforms, camera_centers

def construct_list_of_attributes():
    # taken from gaussian splatting repo. 
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    # 3 channels for DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    # 9 channels for SH order 1
    for i in range(9):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l

@torch.no_grad()
def export_to_obj(reconstruction, ply_out_path):
    """
    Args:
      reconstruction: dict with xyz, opacity, features dc, etc with leading batch size
      ply_out_path: file path where to save the output
    """
    os.makedirs(os.path.dirname(ply_out_path), exist_ok=True)

    for k, v in reconstruction.items():
        # check dimensions
        if k not in ["features_dc", "features_rest"]:
            assert len(v.shape) == 3, "Unexpected size for {}".format(k)
        else:
            assert len(v.shape) == 4, "Unexpected size for {}".format(k)
        assert v.shape[0] == 1, "Expected batch size to be 0"
        reconstruction[k] = v[0]

    valid_gaussians = torch.where(reconstruction["opacity"] > -2.5)[0]

    # transforms for visualisation in Gradio
    # ============= Transform locations =============
    xyz = reconstruction["xyz"][valid_gaussians].detach().cpu().clone()
    t1 = torch.tensor([[1,  0, 0],
                        [0,  0, 1],
                        [0, -1, 0]], dtype=torch.float32)
    angle1 = 30 * math.pi * 2 / 360
    t2 = torch.tensor([[math.cos(angle1),  -math.sin(angle1), 0],
                       [math.sin(angle1),  math.cos(angle1), 0],
                       [0, 0, 1]], dtype=torch.float32)
    angle2 = -60 * math.pi * 2 / 360
    t3 = torch.tensor([[math.cos(angle2), 0, math.sin(angle2)],
                       [0,  1, 0],
                       [-math.sin(angle2), 0, math.cos(angle2)]], dtype=torch.float32)

    overall_transform_matrix = (t1 @ t2)@ t3

    xyz = torch.matmul(xyz, overall_transform_matrix).numpy()
    normals = np.zeros_like(xyz)

    # ============= Transform rotations =============
    camera_transformation_matrix = overall_transform_matrix.inverse()
    camera_quaternions = matrix_to_quaternion(camera_transformation_matrix).to(reconstruction["rotation"].device)
    rotation = reconstruction["rotation"].clone()
    rotation = rotation.unsqueeze(0)
    rotation = quaternion_raw_multiply(camera_quaternions.unsqueeze(0).unsqueeze(0).expand(*rotation.shape), 
                rotation).squeeze(0)

    f_dc = reconstruction["features_dc"][valid_gaussians].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = reconstruction["features_rest"][valid_gaussians].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = reconstruction["opacity"][valid_gaussians].detach().contiguous().cpu().numpy()

    # enlarge Gaussians - otherwise transforming them to .ply results in artefacts
    scale = (reconstruction["scaling"][valid_gaussians] + torch.abs(reconstruction["scaling"][valid_gaussians] * 0.1)).detach().contiguous().cpu().numpy()
    rotation = rotation[valid_gaussians].detach().contiguous().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]  
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_out_path)
