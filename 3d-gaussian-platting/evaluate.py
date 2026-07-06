import torch
import numpy as np
from pytorch_msssim import ssim
import lpips

def generate_novel_viewpoints(
    base_intrinsics: dict,
    base_extrinsics: dict,
    num_views: int = 10,
    radius: float = 1.0,
    elevation: float = 0.0,
    center_point: np.array = np.array([0.0, 0.0, 0.0])
) -> list:
    """
    Generates camera parameters for novel viewpoints by rotating around a central point.
    """
    novel_view_params = []

    for i in range(num_views):
        angle_rad = 2 * np.pi * i / num_views

        x = center_point[0] + radius * np.sin(angle_rad)
        z = center_point[2] + radius * np.cos(angle_rad)
        y = center_point[1] + elevation
        new_cam_pos_world = np.array([x, y, z])

        look_at_vector = center_point - new_cam_pos_world
        look_at_vector = look_at_vector / np.linalg.norm(look_at_vector)

        camera_z_axis = look_at_vector
        world_up = np.array([0.0, 1.0, 0.0])
        if np.abs(np.dot(camera_z_axis, world_up)) > 0.99:
            world_up = np.array([0.0, 0.0, 1.0])

        camera_x_axis = np.cross(world_up, camera_z_axis)
        camera_x_axis = camera_x_axis / (np.linalg.norm(camera_x_axis) + 1e-8)
        camera_y_axis = np.cross(camera_z_axis, camera_x_axis)
        camera_y_axis = camera_y_axis / (np.linalg.norm(camera_y_axis) + 1e-8)

        new_R_world_to_cam = np.vstack([camera_x_axis, camera_y_axis, camera_z_axis])
        new_T_world_to_cam = -new_R_world_to_cam @ new_cam_pos_world.reshape(3, 1)

        novel_view_params.append({
            'intrinsics': base_intrinsics,
            'extrinsics': {
                'rotation_matrix': new_R_world_to_cam,
                'translation_vector': new_T_world_to_cam
            }
        })

    # print(f"Generated {num_views} novel viewpoints.")
    return novel_view_params

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    if img1.device != img2.device:
        img2 = img2.to(img1.device)
    if img1.dtype != img2.dtype:
        img2 = img2.to(img1.dtype)

    mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return torch.tensor(float('inf'))

    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Calculates the Structural Similarity Index Measure (SSIM) between two images.
    """
    if img1.device != img2.device:
        img2 = img2.to(img1.device)
    if img1.dtype != img2.dtype:
        img2 = img2.to(img1.dtype)

    img1_nchw = img1.permute(2, 0, 1).unsqueeze(0)
    img2_nchw = img2.permute(2, 0, 1).unsqueeze(0)

    ssim_val = ssim(img1_nchw, img2_nchw, data_range=data_range, size_average=True)
    return ssim_val

def calculate_lpips(img1: torch.Tensor, img2: torch.Tensor, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Calculates the Learned Perceptual Image Patch Similarity (LPIPS) between two images.
    """
    loss_fn_alex = lpips.LPIPS(net='alex', spatial=False).to(device)
    loss_fn_alex.eval()

    img1 = img1.to(device).float()
    img2 = img2.to(device).float()

    img1_nchw = img1.permute(2, 0, 1).unsqueeze(0)
    img2_nchw = img2.permute(2, 0, 1).unsqueeze(0)

    img1_lpips_format = img1_nchw * 2 - 1
    img2_lpips_format = img2_nchw * 2 - 1

    with torch.no_grad():
        lpips_value = loss_fn_alex(img1_lpips_format, img2_lpips_format)

    return lpips_value.squeeze()
