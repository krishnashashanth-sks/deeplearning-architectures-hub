import torch
import numpy as np
from scipy.spatial import KDTree
from layers import quaternion_to_rotation_matrix

def prune_gaussians(
    gradient_gaussian_parameters: dict,
    min_opacity_threshold: float = 0.005,
    min_scale_threshold: float = 0.001
) -> dict:
    """
    Prunes Gaussians based on their opacity and scale.
    """
    print("\n--- Pruning Gaussians ---")
    current_num_gaussians = gradient_gaussian_parameters['means'].shape[0]
    if current_num_gaussians == 0:
        print("No Gaussians to prune.")
        return gradient_gaussian_parameters

    opacities = gradient_gaussian_parameters['opacities'].detach()
    scales = torch.exp(gradient_gaussian_parameters['scales'].detach()) 

    keep_mask_opacity = (opacities > min_opacity_threshold).squeeze(-1)
    avg_scales = scales.mean(dim=1)
    keep_mask_scale = (avg_scales > min_scale_threshold)

    keep_mask = keep_mask_opacity & keep_mask_scale

    updated_params = {}
    if keep_mask.sum() == 0: # If all Gaussians are pruned, create a minimal set to avoid empty tensors
        print("Warning: All Gaussians pruned. Re-initializing with a single dummy Gaussian.")
        dummy_mean = torch.tensor([[0.0, 0.0, 0.0]], device=opacities.device, dtype=torch.float32)
        dummy_scale = torch.tensor([[-2.0, -2.0, -2.0]], device=opacities.device, dtype=torch.float32) # log-scale
        dummy_rotation = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=opacities.device, dtype=torch.float32)
        dummy_opacity = torch.tensor([[0.1]], device=opacities.device, dtype=torch.float32)
        dummy_sh_coeffs = torch.zeros(1, 16, 3, device=opacities.device, dtype=torch.float32)
        dummy_sh_coeffs[:, 0, :] = 0.5 # Mid-gray color

        updated_params['means'] = torch.nn.Parameter(dummy_mean.requires_grad_(True))
        updated_params['scales'] = torch.nn.Parameter(dummy_scale.requires_grad_(True))
        updated_params['rotations'] = torch.nn.Parameter(dummy_rotation.requires_grad_(True))
        updated_params['opacities'] = torch.nn.Parameter(dummy_opacity.requires_grad_(True))
        updated_params['sh_coeffs'] = torch.nn.Parameter(dummy_sh_coeffs.requires_grad_(True))
    else:
        for key, param_tensor in gradient_gaussian_parameters.items():
            updated_params[key] = torch.nn.Parameter(param_tensor[keep_mask])

    pruned_num_gaussians = updated_params['means'].shape[0]
    print(f"Pruned {current_num_gaussians - pruned_num_gaussians} Gaussians.")
    print(f"New number of Gaussians: {pruned_num_gaussians}")
    print("-------------------------")

    return updated_params

def split_gaussians(
    gradient_gaussian_parameters: dict,
    position_grad_threshold: float = 0.0002,
    scale_threshold: float = 0.01,
    split_factor: float = 0.8,
    opacity_factor: float = 0.5,
) -> dict:
    """
    Splits Gaussians that have high position gradients or are too large.
    """
    print("\n--- Splitting Gaussians ---")
    current_num_gaussians = gradient_gaussian_parameters['means'].shape[0]
    if current_num_gaussians == 0:
        print("No Gaussians to split.")
        return gradient_gaussian_parameters

    means = gradient_gaussian_parameters['means'].detach()
    scales = gradient_gaussian_parameters['scales'].detach()
    rotations = gradient_gaussian_parameters['rotations'].detach()
    opacities = gradient_gaussian_parameters['opacities'].detach()
    sh_coeffs = gradient_gaussian_parameters['sh_coeffs'].detach()

    split_mask_grad = torch.zeros(current_num_gaussians, dtype=torch.bool, device=means.device)
    if means.grad is not None:
        mean_grads = means.grad.norm(dim=1)
        split_mask_grad = (mean_grads > position_grad_threshold)

    linear_scales = torch.exp(scales)
    avg_linear_scales = linear_scales.mean(dim=1)
    split_mask_scale = (avg_linear_scales > scale_threshold)

    split_mask = split_mask_grad | split_mask_scale

    keep_indices = ~split_mask
    split_indices = split_mask

    new_means = [means[keep_indices]]
    new_scales = [scales[keep_indices]]
    new_rotations = [rotations[keep_indices]]
    new_opacities = [opacities[keep_indices]]
    new_sh_coeffs = [sh_coeffs[keep_indices]]

    gaussians_to_split_count = split_indices.sum().item()
    if gaussians_to_split_count > 0:
        split_means = means[split_indices]
        split_scales = scales[split_indices]
        split_rotations = rotations[split_indices]
        split_opacities = opacities[split_indices]
        split_sh_coeffs = sh_coeffs[split_indices]

        for i in range(gaussians_to_split_count):
            mean = split_means[i:i+1]
            scale = split_scales[i:i+1]
            rotation = split_rotations[i:i+1]
            opacity = split_opacities[i:i+1]
            sh_coeff = split_sh_coeffs[i:i+1]

            linear_scale_orig = torch.exp(scale.squeeze())
            R_gaussian = quaternion_to_rotation_matrix(rotation)

            perturb_vector = torch.zeros_like(mean)
            max_scale_idx = torch.argmax(linear_scale_orig)
            perturb_amount = linear_scale_orig[max_scale_idx] * 0.2
            perturb_vector[0, max_scale_idx] = perturb_amount

            perturbed_dir = torch.matmul(R_gaussian.squeeze(0), perturb_vector.T).T

            new_mean1 = mean + perturbed_dir
            new_mean2 = mean - perturbed_dir

            new_scale = torch.log(torch.exp(scale) * split_factor)
            new_opacity = opacity * opacity_factor

            new_means.extend([new_mean1, new_mean2])
            new_scales.extend([new_scale, new_scale])
            new_rotations.extend([rotation, rotation])
            new_opacities.extend([new_opacity, new_opacity])
            new_sh_coeffs.extend([sh_coeff, sh_coeff])

    updated_params = {
        'means': torch.nn.Parameter(torch.cat(new_means, dim=0).requires_grad_(True)),
        'scales': torch.nn.Parameter(torch.cat(new_scales, dim=0).requires_grad_(True)),
        'rotations': torch.nn.Parameter(torch.cat(new_rotations, dim=0).requires_grad_(True)),
        'opacities': torch.nn.Parameter(torch.cat(new_opacities, dim=0).requires_grad_(True)),
        'sh_coeffs': torch.nn.Parameter(torch.cat(new_sh_coeffs, dim=0).requires_grad_(True)),
    }

    new_num_gaussians = updated_params['means'].shape[0]
    print(f"Split {gaussians_to_split_count} Gaussians into {gaussians_to_split_count * 2} new ones.")
    print(f"New number of Gaussians: {new_num_gaussians}")
    print("-------------------------")

    return updated_params

def initialize_gaussians(points, colors):
    """
    Initializes 3D Gaussian parameters (mean, covariance, opacity, spherical harmonics)
    from an initial point cloud.

    Args:
        points (np.array): Nx3 array of point coordinates.
        colors (np.array): Nx3 array of point colors (RGB, 0-255).

    Returns:
        dict: A dictionary containing initialized Gaussian parameters as PyTorch tensors:
              'means', 'scales', 'rotations', 'opacities', 'sh_coeffs'.
    """
    num_points = points.shape[0]

    # 1. Initialize mean (position)
    means = torch.from_numpy(points).float()

    # 2. Convert colors to [0, 1] range and use as L0 spherical harmonic coefficients
    colors_normalized = torch.from_numpy(colors / 255.0).float()

    max_sh_degree = 3 # This can be made configurable
    num_sh_coefficients_per_channel = (max_sh_degree + 1) ** 2
    sh_coeffs = torch.zeros(num_points, num_sh_coefficients_per_channel, 3, dtype=torch.float32)
    sh_coeffs[:, 0, :] = colors_normalized # Set L0 coefficients

    # 3. Initialize opacity
    opacities = torch.full((num_points, 1), 0.1, dtype=torch.float32)

    # 5. Initialize scales based on average distance to 3 nearest neighbors
    if num_points > 1:
        kdtree = KDTree(points)
        # Query for k=4 because the point itself will be included as the 0th neighbor
        distances, _ = kdtree.query(points, k=4)
        # The first column is distance to itself (0), so we take the next 3 neighbors
        avg_distances = np.mean(distances[:, 1:4], axis=1, keepdims=True)
        # Initialize scales isotropically with this average distance
        scales = torch.from_numpy(np.tile(avg_distances, (1, 3))).float()
    else:
        # Handle case with a single point to avoid KDTree error
        scales = torch.full((num_points, 3), 0.01, dtype=torch.float32) # Default small scale

    # Apply a small log-scale to avoid zero scales and allow gradient flow
    scales = torch.log(scales + 1e-6)

    # 6. Initialize rotation to identity quaternion
    # Quaternion representation (x, y, z, w) where (0,0,0,1) is identity
    rotations = torch.full((num_points, 4), 0.0, dtype=torch.float32)
    rotations[:, 3] = 1.0 # Set 'w' component to 1 for identity quaternion

    return {
        'means': means,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacities,
        'sh_coeffs': sh_coeffs
    }

