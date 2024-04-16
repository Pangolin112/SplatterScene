import os
import glob
import json
import math
import torch
import torchvision
import numpy as np

from PIL import Image

from .objaverse import ObjaverseDataset

from utils.graphics_utils import getProjectionMatrix

GSO_ROOT = None # Change this to your data directory
assert GSO_ROOT is not None, "Update path of the dataset"

class GSODataset(ObjaverseDataset):
    def __init__(self,
                 cfg,
                 dataset_name = "test",
                 ) -> None:
        
        super(GSODataset).__init__()

        self.cfg = cfg
        self.root_dir = GSO_ROOT
        assert dataset_name != "train", "No training on GSO dataset!"

        self.dataset_name = dataset_name

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)

        self.image_side_target = self.cfg.data.training_resolution
        self.opengl_to_colmap = torch.tensor([[  1,  0,  0,  0],
                                              [  0, -1,  0,  0],
                                              [  0,  0, -1,  0],
                                              [  0,  0,  0,  1]], dtype=torch.float32)

        self.paths = sorted(
            glob.glob(os.path.join(self.root_dir, "*"))
        )

        print('============= length of dataset %d =============' % len(self.paths))

        self.test_input_idxs = [0]

    def __len__(self):
        return len(self.paths)

    def get_example_id(self, index):
        example_path = self.paths[index]
        return os.path.basename(example_path)

    def __getitem__(self, index):
        filename = self.paths[index]
        paths = sorted(glob.glob(os.path.join(filename, 'render_mvs_25/model/*.png')))

        if self.dataset_name == "vis":
            images_and_camera_poses = self.load_loop(paths, 100)
        else:
            images_and_camera_poses = self.load_imgs_and_convert_cameras(paths, len(paths))

        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])
        
        return images_and_camera_poses