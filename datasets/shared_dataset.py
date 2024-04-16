import torch
from torch.utils.data import Dataset

from utils.general_utils import matrix_to_quaternion

class SharedDataset(Dataset):
    """
    Parent dataset class with shared functions
    """
    def __init__(self) -> None:
        super().__init__()

    def make_poses_relative_to_first(self, images_and_camera_poses):
        inverse_first_camera = images_and_camera_poses["world_view_transforms"][0].inverse().clone()
        for c in range(images_and_camera_poses["world_view_transforms"].shape[0]):
            images_and_camera_poses["world_view_transforms"][c] = torch.bmm(
                                                inverse_first_camera.unsqueeze(0),
                                                images_and_camera_poses["world_view_transforms"][c].unsqueeze(0)).squeeze(0)
            images_and_camera_poses["view_to_world_transforms"][c] = torch.bmm(
                                                images_and_camera_poses["view_to_world_transforms"][c].unsqueeze(0),
                                                inverse_first_camera.inverse().unsqueeze(0)).squeeze(0)
            images_and_camera_poses["full_proj_transforms"][c] = torch.bmm(
                                                inverse_first_camera.unsqueeze(0),
                                                images_and_camera_poses["full_proj_transforms"][c].unsqueeze(0)).squeeze(0)
            images_and_camera_poses["camera_centers"][c] = images_and_camera_poses["world_view_transforms"][c].inverse()[3, :3]
        return images_and_camera_poses
    
    def get_source_cw2wT(self, source_cameras_view_to_world):
        # Compute view to world transforms in quaternion representation.
        # Used for transforming predicted rotations
        qs = []
        for c_idx in range(source_cameras_view_to_world.shape[0]):
            qs.append(matrix_to_quaternion(source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)))
        return torch.stack(qs, dim=0)