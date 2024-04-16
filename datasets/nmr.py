import os
import shutil
import subprocess
import tempfile
import torch
import torch.nn.functional as F
import glob
from PIL import Image
import numpy as np
import zipfile

from utils.general_utils import PILtoTorch
from utils.graphics_utils import getProjectionMatrix
from utils.camera_utils import get_loop_cameras

from .shared_dataset import SharedDataset

NMR_DATASET_ROOT = None # Change this to your data directory
assert NMR_DATASET_ROOT is not None, "Update path of the dataset"

class NMRDataset(SharedDataset):
    """
    Dataset from DVR (Niemeyer et al. 2020)
    Provides NMR renderings
    """

    def __init__(self, cfg, 
                 dataset_name="train"
                 ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name

        # first check if the dataset is already on the local machine
        self.base_path = NMR_DATASET_ROOT

        list_prefix = "softras_"

        # list all categories
        cats = [x for x in glob.glob(os.path.join(self.base_path, "*")) if os.path.isdir(x)]

        # Read the split for each category
        if dataset_name == "train":
            file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        elif dataset_name == "val" or dataset_name == "vis":
            file_lists = [os.path.join(x, list_prefix + "val.lst") for x in cats]
        elif dataset_name == "test":
            file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]

        # Create a list of all objects in the split across categories
        all_objs = []
        num_objs = []

        # For comparison against baselines on the test subset read the required views
        if dataset_name == "test":
            # src viewlist was copied from
            # https://github.com/sxyu/pixel-nerf/blob/91a044bdd62aebe0ed3a5685ca37cb8a9dc8e8ee/viewlist/src_dvr.txt
            source_view_fname =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                              "viewlist_src_nmr.txt")
            with open(source_view_fname, "r") as f:
                src_view_list = f.readlines()
            self.src_view_dict = {}
            for line in src_view_list:
                cat, obj_idx, src_view_idx = line.split()
                if cat not in self.src_view_dict.keys():
                    self.src_view_dict[cat] = {}
                self.src_view_dict[cat][obj_idx] = int(src_view_idx)
                obj = (cat, os.path.join(self.base_path, cat, obj_idx))
                all_objs.append(obj)
            print("found {} objs".format(len(all_objs)))
        else:
            for file_list in file_lists:
                if not os.path.exists(file_list):
                    continue
                base_dir = os.path.dirname(file_list)
                cat = os.path.basename(base_dir)
                with open(file_list, "r") as f:
                    objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
                all_objs.extend(objs)
                num_objs.append(len(objs))

            print("Found {} categories, with {} files in them, respecively".format(
                len(cats), num_objs))

        self.all_objs = all_objs

        self._coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        self._coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )

        self._pixelnerf_to_colmap = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self.imgs_per_obj_train = self.cfg.opt.imgs_per_obj

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)

        if cfg.data.subset != -1:
            self.paths = self.paths[:cfg.data.subset]

    def load_imgs_and_convert_cameras(self, rgb_paths, cam_path, num_views):
        """
        Load the images, camera matrices and projection matrices for a given object 
        """
        world_view_transforms = []
        view_world_transforms = []

        camera_centers = []
        imgs = []

        all_cam = np.load(cam_path)

        # validation dataset is used for scoring - fix cond frame for reproducibility
        # in trainng need to randomly sample the conditioning frame
        if self.dataset_name == "val": 
            indexes = torch.arange(num_views)
        elif self.dataset_name == "test":
            cat_id, obj_id, _ = cam_path.split("/")[-3:]
            src_idx = self.src_view_dict[cat_id][obj_id]
            indexes = torch.tensor([src_idx] + [i for i in range(len(rgb_paths)) if i != src_idx])
        else:
            indexes = torch.randperm(len(rgb_paths))[:num_views]
            indexes = torch.cat([indexes[:self.cfg.data.input_images], indexes], dim=0)

        for frame_idx in indexes:

            rgb_path = rgb_paths[frame_idx]
            img = PILtoTorch(Image.open(rgb_path), (self.cfg.data.training_resolution,
                                                    self.cfg.data.training_resolution))
            imgs.append(img)

            # Read off extrinsic matrix
            wmat_inv_key = "world_mat_inv_" + str(frame_idx.item())
            wmat_key = "world_mat_" + str(frame_idx.item())
            if wmat_inv_key in all_cam:
                c2w_cmo = all_cam[wmat_inv_key]
            else:
                w2c_cmo = all_cam[wmat_key]
                if w2c_cmo.shape[0] == 3:
                    w2c_cmo = np.vstack((w2c_cmo, np.array([0, 0, 0, 1])))
                c2w_cmo = np.linalg.inv(w2c_cmo)

            # focal length in the dataset is in NDC units
            # check that focal length is the same in all loaded files
            # fov for projection matrix was pre-computed
            intr_mtx = all_cam["camera_mat_" + str(frame_idx.item())]
            fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
            assert abs(fx - 3.7320509) < 1e-5, "Different focal length found"
            assert abs(fx - fy) < 1e-9

            # pose in pixelnerf coordinate system
            # PixelNeRF coordinate system is (x right, y up, z into camera)
            # need to change to COLMAP system (x right y down z forward)
            c2w_cmo = (
                self._coord_trans_world
                @ torch.tensor(c2w_cmo, dtype=torch.float32)
                @ self._coord_trans_cam # to pixelnerf coordinate system
                @ self._pixelnerf_to_colmap # to colmap coordinate system
            ) 

            # put in row-major order
            view_world_transform = c2w_cmo.transpose(0, 1)
            world_view_transform = c2w_cmo.inverse().transpose(0, 1)
            camera_center = view_world_transform[3, :3].clone()

            world_view_transforms.append(world_view_transform)
            view_world_transforms.append(view_world_transform)
            camera_centers.append(camera_center)

        imgs = torch.stack(imgs)
        world_view_transforms = torch.stack(world_view_transforms)
        view_world_transforms = torch.stack(view_world_transforms)
        camera_centers = torch.stack(camera_centers)

        full_proj_transforms = world_view_transforms.bmm(self.projection_matrix.unsqueeze(0).expand(
            world_view_transforms.shape[0], 4, 4))

        return {"gt_images": imgs,
                "world_view_transforms": world_view_transforms,
                "view_to_world_transforms": view_world_transforms,
                "full_proj_transforms": full_proj_transforms,
                "camera_centers": camera_centers}

    def load_loop(self, rgb_paths, cam_path, num_imgs_in_loop):
        world_view_transforms = []
        view_world_transforms = []
        camera_centers = []
        imgs = []

        gt_imgs_and_cameras = self.load_imgs_and_convert_cameras(rgb_paths, cam_path, len(rgb_paths))
        loop_cameras_c2w_cmo = get_loop_cameras(num_imgs_in_loop=num_imgs_in_loop, radius=2.73)

        for src_idx in range(self.cfg.data.input_images):
            imgs.append(gt_imgs_and_cameras["gt_images"][src_idx])
            camera_centers.append(gt_imgs_and_cameras["camera_centers"][src_idx])
            world_view_transforms.append(gt_imgs_and_cameras["world_view_transforms"][src_idx])
            view_world_transforms.append(gt_imgs_and_cameras["view_to_world_transforms"][src_idx])

        for loop_camera_c2w_cmo in loop_cameras_c2w_cmo:
            view_world_transform = torch.from_numpy(loop_camera_c2w_cmo).transpose(0, 1)
            world_view_transform = torch.from_numpy(loop_camera_c2w_cmo).inverse().transpose(0, 1)
            camera_center = view_world_transform[3, :3].clone()

            camera_centers.append(camera_center)
            world_view_transforms.append(world_view_transform)
            view_world_transforms.append(view_world_transform)

            # use the closest camera as reference gt image
            closest_gt_idx = torch.argmin(torch.norm(
                gt_imgs_and_cameras["camera_centers"] - camera_center.unsqueeze(0), dim=-1)).item()
            imgs.append(gt_imgs_and_cameras["gt_images"][closest_gt_idx])

        imgs = torch.stack(imgs)
        world_view_transforms = torch.stack(world_view_transforms)
        view_world_transforms = torch.stack(view_world_transforms)
        camera_centers = torch.stack(camera_centers)

        full_proj_transforms = world_view_transforms.bmm(self.projection_matrix.unsqueeze(0).expand(
            world_view_transforms.shape[0], 4, 4))

        return {"gt_images": imgs,
                "world_view_transforms": world_view_transforms,
                "view_to_world_transforms": view_world_transforms,
                "full_proj_transforms": full_proj_transforms,
                "camera_centers": camera_centers}

    def __len__(self):
        return len(self.all_objs)

    def get_example_id(self, index):
        cat, root_dir = self.all_objs[index]
        return cat + "_" + os.path.basename(root_dir)

    def __getitem__(self, index):
        _, root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob.glob(os.path.join(root_dir, "image", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]
        rgb_paths = sorted(rgb_paths)

        cam_path = os.path.join(root_dir, "cameras.npz")

        if self.dataset_name == "train":
            num_views = self.imgs_per_obj_train
        else:
            num_views = len(rgb_paths)

        if self.dataset_name == "vis":
            images_and_camera_poses = self.load_loop(rgb_paths, cam_path, 100)    
        else:
            images_and_camera_poses = self.load_imgs_and_convert_cameras(rgb_paths, cam_path, num_views)

        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses