import glob
import os

from einops import repeat

import numpy as np
import torch
from torch.utils.data import Dataset
import math

from .dataset_readers import readCamerasFromTxt
from utils.general_utils import PILtoTorch, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World, fov2focal

from .shared_dataset import SharedDataset
from . import base_path as bp

SHAPENET_DATASET_ROOT = bp.SHAPENET_DATASET_ROOT
assert SHAPENET_DATASET_ROOT is not None, "Update the location of the SRN Shapenet Dataset"

class SRNDataset(SharedDataset):
    def __init__(self, cfg,
                 dataset_name="train"):
        super().__init__()
        self.cfg = cfg

        self.dataset_name = dataset_name
        if dataset_name == "vis":
            self.dataset_name = "test"

        self.base_path = os.path.join(SHAPENET_DATASET_ROOT, "srn_{}/{}_{}".format(cfg.data.category,
                                                                                   cfg.data.category,
                                                                                   self.dataset_name))

        is_chair = "chair" in cfg.data.category
        if is_chair and dataset_name == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )

        print(len(self.intrins))
        if cfg.data.subset != -1:
            self.intrins = self.intrins[:cfg.data.subset]

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=cfg.data.fovx * 2 * np.pi / 360,
            fovY=cfg.data.fovy * 2 * np.pi / 360).transpose(0, 1)
        
        self.imgs_per_obj = self.cfg.opt.imgs_per_obj

        # in deterministic version the number of testing images
        # and number of training images are the same
        if self.cfg.data.input_images == 1:
            self.test_input_idxs = [7]
        elif self.cfg.data.input_images == 2:
            self.test_input_idxs = [4, 12]
        else:
            raise NotImplementedError

        self.init_ray_dirs()

    def __len__(self):
        return len(self.intrins)

    def init_ray_dirs(self):
        """
        Initialises an image of ray directions, unscaled by focal lengths.
        """
        x = torch.linspace(-63.5, 63.5, 128)
        y = torch.linspace(63.5, -63.5, 128)
        if self.cfg.model.inverted_x:
            x = -x
        if self.cfg.model.inverted_y:
            y = -y
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        ones = torch.ones_like(grid_x, dtype=grid_x.dtype)
        ray_dirs = torch.stack([grid_x, grid_y, ones]).unsqueeze(0)
        self.ray_dirs = ray_dirs

    def get_origin_distance(self, camera_to_world):
        # outputs the origin_distances. This helps resolve depth
        # ambiguity in single-view depth estimation. Follows PixelNeRF
        camera_center_to_origin = - camera_to_world[3, :3]
        camera_z_vector = camera_to_world[2, :3]
        origin_distances = torch.dot(camera_center_to_origin, camera_z_vector).unsqueeze(0)
        origin_distances = repeat(origin_distances, 'c -> c h w',
                                  h=self.cfg.data.training_resolution, w=self.cfg.data.training_resolution)

        return origin_distances

    def load_example_id(self, example_id, intrin_path,
                        trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        #pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        ############ for depth #################################
        pose_colmap_depth_paths = sorted(glob.glob(os.path.join(dir_path, "pose_colmap_depth", "*")))
        depth_paths = sorted(glob.glob(os.path.join(dir_path, "depth", "*")))
        ############ for depth #################################

        assert len(rgb_paths) == len(pose_colmap_depth_paths) == len(depth_paths)

        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}

            self.all_depths = {}
            self.all_colmap_depth_Rs = {}
            self.all_colmap_depth_Ts = {}
            self.all_Ks = {}

            self.all_focals_pixels = {}
            self.all_origin_distances = {}
            self.all_ray_embeddings = {}

        if example_id not in self.all_rgbs.keys():
            self.all_rgbs[example_id] = []
            self.all_world_view_transforms[example_id] = []
            self.all_full_proj_transforms[example_id] = []
            self.all_camera_centers[example_id] = []
            self.all_view_to_world_transforms[example_id] = []

            self.all_depths[example_id] = []
            self.all_colmap_depth_Rs[example_id] = []
            self.all_colmap_depth_Ts[example_id] = []
            self.all_Ks[example_id] = []

            self.all_focals_pixels[example_id] = []
            self.all_origin_distances[example_id] = []
            self.all_ray_embeddings[example_id] = []

            cam_infos = readCamerasFromTxt(rgb_paths, pose_colmap_depth_paths, depth_paths, [i for i in range(len(rgb_paths))])

            for cam_info in cam_infos:
                R = cam_info.R
                T = cam_info.T

                self.all_rgbs[example_id].append(PILtoTorch(cam_info.image, 
                                                            (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])

                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                self.all_world_view_transforms[example_id].append(world_view_transform)
                self.all_view_to_world_transforms[example_id].append(view_world_transform)
                self.all_full_proj_transforms[example_id].append(full_proj_transform)
                self.all_camera_centers[example_id].append(camera_center)

                ############ for depth #################################
                self.all_depths[example_id].append(PILtoTorch(cam_info.depth, (self.cfg.data.training_resolution, self.cfg.data.training_resolution)))
                K = np.array([[30.0, 0, 64],  # stupid hardcoded values
                              [0, 50.0, 64],
                              [0, 0, 1]], dtype=np.float32)
                self.all_Ks[example_id].append(torch.from_numpy(K))
                R_colmap_depth = cam_info.R_colmap_depth
                T_colmap_depth = cam_info.T_colmap_depth
                self.all_colmap_depth_Rs[example_id].append(torch.tensor(R_colmap_depth))
                self.all_colmap_depth_Ts[example_id].append(torch.tensor(T_colmap_depth))
                ############ for depth #################################

                self.all_focals_pixels[example_id].append(
                    torch.tensor([fov2focal(cam_info.FovX, 128), fov2focal(cam_info.FovY, 128)]))
                self.all_origin_distances[example_id].append(
                    self.get_origin_distance(self.all_view_to_world_transforms[example_id][-1]))
                ray_dirs = self.ray_dirs.clone()[0]
                ray_dirs[:2, ...] = ray_dirs[:2, ...] / self.all_focals_pixels[example_id][-1].unsqueeze(1).unsqueeze(2)
                self.all_ray_embeddings[example_id].append(ray_dirs)

            self.all_world_view_transforms[example_id] = torch.stack(self.all_world_view_transforms[example_id])
            self.all_view_to_world_transforms[example_id] = torch.stack(self.all_view_to_world_transforms[example_id])
            self.all_full_proj_transforms[example_id] = torch.stack(self.all_full_proj_transforms[example_id])
            self.all_camera_centers[example_id] = torch.stack(self.all_camera_centers[example_id])
            self.all_rgbs[example_id] = torch.stack(self.all_rgbs[example_id])

            ############ for depth #################################
            self.all_depths[example_id] = torch.stack(self.all_depths[example_id])
            self.all_Ks[example_id] = torch.stack(self.all_Ks[example_id])
            self.all_colmap_depth_Rs[example_id] = torch.stack(self.all_colmap_depth_Rs[example_id])
            self.all_colmap_depth_Ts[example_id] = torch.stack(self.all_colmap_depth_Ts[example_id])
            ############ for depth #################################

            self.all_focals_pixels[example_id] = torch.stack(self.all_focals_pixels[example_id])
            self.all_origin_distances[example_id] = torch.stack(self.all_origin_distances[example_id])
            self.all_ray_embeddings[example_id] = torch.stack(self.all_ray_embeddings[example_id])

    def get_example_id(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        return example_id

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))

        self.load_example_id(example_id, intrin_path)
        if self.dataset_name == "train":
            frame_idxs = torch.randperm(
                    len(self.all_rgbs[example_id])
                    )[:self.imgs_per_obj]

            frame_idxs = torch.cat([frame_idxs[:self.cfg.data.input_images], frame_idxs], dim=0)

        else:
            input_idxs = self.test_input_idxs

            frame_idxs = torch.cat([torch.tensor(input_idxs), torch.tensor([i for i in range(15) if i not in input_idxs])], dim=0)

        images_and_camera_poses = {
            "ray_embeddings": self.all_ray_embeddings[example_id][frame_idxs],
            "origin_distances": self.all_origin_distances[example_id][frame_idxs],
            "focals_pixels": self.all_focals_pixels[example_id][frame_idxs].clone(),
            "gt_images": self.all_rgbs[example_id][frame_idxs].clone(),
            "world_view_transforms": self.all_world_view_transforms[example_id][frame_idxs],
            "view_to_world_transforms": self.all_view_to_world_transforms[example_id][frame_idxs],
            "full_proj_transforms": self.all_full_proj_transforms[example_id][frame_idxs],
            "camera_centers": self.all_camera_centers[example_id][frame_idxs],
            "gt_depths": self.all_depths[example_id][frame_idxs].clone(),
            "Ks": self.all_Ks[example_id][frame_idxs],
            "colmap_depth_Rs": self.all_colmap_depth_Rs[example_id][frame_idxs],
            "colmap_depth_Ts": self.all_colmap_depth_Ts[example_id][frame_idxs]
        }

        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses