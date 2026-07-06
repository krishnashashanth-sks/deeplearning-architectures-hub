import torch
from layers import quaternion_to_rotation_matrix,evaluate_sh_basis

def render_gaussian(
    means: torch.Tensor, # Nx3 tensor
    scales: torch.Tensor, # Nx3 tensor
    rotations: torch.Tensor, # Nx4 tensor (quaternions)
    opacities: torch.Tensor, # Nx1 tensor
    sh_coeffs: torch.Tensor, # Nx(num_sh_coeffs)x3 tensor
    camera_intrinsics: dict, # {'focal_length': (fx, fy), 'principal_point': (cx, cy)}
    camera_extrinsics: dict, # {'rotation_matrix': 3x3, 'translation_vector': 3x1}
    image_dimensions: tuple # (height, width)
) -> torch.Tensor:
    """
    Renders a 2D image from 3D Gaussian parameters given camera parameters.

    Args:
        means: N x 3 tensor of Gaussian means (3D positions).
        scales: N x 3 tensor of Gaussian scales (log scale).
        rotations: N x 4 tensor of Gaussian rotations (quaternions, xyzw).
        opacities: N x 1 tensor of Gaussian opacities.
        sh_coeffs: N x M x 3 tensor of spherical harmonic coefficients for RGB colors.
        camera_intrinsics: Dictionary containing focal length and principal point.
        camera_extrinsics: Dictionary containing camera rotation matrix and translation vector.
        image_dimensions: Tuple (height, width) of the target rendered image.

    Returns:
        A H x W x 3 PyTorch tensor representing the rendered image.
    """
    height, width = image_dimensions
    device = means.device

    rendered_image = torch.zeros((height, width, 3), device=device, dtype=torch.float32)
    alpha_accumulated_transmittance = torch.ones((height, width, 1), device=device, dtype=torch.float32)

    num_gaussians = means.shape[0]
    if num_gaussians == 0:
        return rendered_image # Return empty black image if no Gaussians

    # --- Step 2: 3D to 2D Projection ---
    R_cam_np = camera_extrinsics['rotation_matrix']
    T_cam_np = camera_extrinsics['translation_vector']

    R_cam = torch.from_numpy(R_cam_np).float().to(device)
    T_cam = torch.from_numpy(T_cam_np).float().to(device)

    if T_cam.shape == (3,):
        T_cam = T_cam.unsqueeze(1) # Ensure it's 3x1

    means_cam = torch.matmul(means, R_cam.T) + T_cam.T

    fx, fy = camera_intrinsics['focal_length']
    cx, cy = camera_intrinsics['principal_point']
    if isinstance(fx, torch.Tensor): fx = fx.item()
    if isinstance(fy, torch.Tensor): fy = fy.item()
    if isinstance(cx, torch.Tensor): cx = cx.item()
    if isinstance(cy, torch.Tensor): cy = cy.item()

    z_cam = means_cam[:, 2:]
    z_cam_safe = torch.where(z_cam > 0.001, z_cam, torch.tensor(0.001, device=device))

    proj_x = fx * (means_cam[:, 0:1] / z_cam_safe) + cx
    proj_y = fy * (means_cam[:, 1:2] / z_cam_safe) + cy

    projected_means_2d = torch.cat([proj_x, proj_y], dim=1)

    # valid_points_mask = (z_cam.squeeze() > 0.001)

    # --- Step 3: Covariance Transformation ---
    linear_scales = torch.exp(scales)
    S_diag = torch.diag_embed(linear_scales)
    R_gaussian = quaternion_to_rotation_matrix(rotations)
    Sigma_world = torch.bmm(R_gaussian, torch.bmm(S_diag, R_gaussian.transpose(1, 2)))

    R_cam_expanded = R_cam.unsqueeze(0).expand(num_gaussians, -1, -1)
    Sigma_cam = torch.bmm(R_cam_expanded, torch.bmm(Sigma_world, R_cam_expanded.transpose(1, 2)))

    X_cam = means_cam[:, 0:1]
    Y_cam = means_cam[:, 1:2]
    Z_cam_jacobian = z_cam_safe

    J = torch.zeros((num_gaussians, 2, 3), device=device, dtype=torch.float32)
    J[:, 0, 0] = fx / Z_cam_jacobian.squeeze()
    J[:, 0, 2] = -fx * X_cam.squeeze() / (Z_cam_jacobian.squeeze()**2)
    J[:, 1, 1] = fy / Z_cam_jacobian.squeeze()
    J[:, 1, 2] = -fy * Y_cam.squeeze() / (Z_cam_jacobian.squeeze()**2)

    Sigma_2D = torch.bmm(J, torch.bmm(Sigma_cam, J.transpose(1, 2)))

    # --- Step 4: Spherical Harmonics Calculation ---
    camera_world_pos = -torch.matmul(R_cam.T, T_cam)
    camera_world_pos_expanded = camera_world_pos.T
    view_dirs = camera_world_pos_expanded - means
    view_dirs_norm = torch.norm(view_dirs, p=2, dim=1, keepdim=True)
    view_dirs_normalized = view_dirs / (view_dirs_norm + 1e-8)

    num_sh_coeffs_per_channel = sh_coeffs.shape[1]
    max_sh_degree = int(torch.sqrt(torch.tensor(num_sh_coeffs_per_channel)).item() - 1)

    sh_basis_values = evaluate_sh_basis(max_sh_degree, view_dirs_normalized)
    view_dependent_colors = torch.sum(sh_coeffs * sh_basis_values.unsqueeze(-1), dim=1)
    view_dependent_colors = torch.clamp(view_dependent_colors, min=0.0, max=1.0)

    # --- Step 5: Depth Sorting ---
    depths = z_cam.squeeze()
    sorted_depths, sorted_indices = torch.sort(depths, descending=False)

    means = means[sorted_indices]
    scales = scales[sorted_indices]
    rotations = rotations[sorted_indices]
    opacities = opacities[sorted_indices]
    sh_coeffs = sh_coeffs[sorted_indices]
    projected_means_2d = projected_means_2d[sorted_indices]
    Sigma_2D = Sigma_2D[sorted_indices]
    view_dependent_colors = view_dependent_colors[sorted_indices]
    means_cam = means_cam[sorted_indices] # Also reorder means_cam for z_cam_safe
    z_cam_safe = z_cam_safe[sorted_indices]

    # --- Step 6: Alpha Blending ---
    pixel_coords_x, pixel_coords_y = torch.meshgrid(
        torch.arange(width, device=device, dtype=torch.float32),
        torch.arange(height, device=device, dtype=torch.float32),
        indexing='xy'
    )
    pixel_coords = torch.stack([pixel_coords_x, pixel_coords_y], dim=-1).reshape(-1, 2) # (H*W, 2)

    for i in range(num_gaussians):
        mean_2d = projected_means_2d[i] # 1x2
        sigma_2d = Sigma_2D[i] # 2x2
        color = view_dependent_colors[i] # 1x3
        opacity_gaussian = opacities[i] # 1x1

        sigma_2d_stable = sigma_2d + torch.eye(2, device=device) * 1e-6
        try:
            sigma_2d_inv = torch.inverse(sigma_2d_stable)
            det_sigma_2d = torch.det(sigma_2d_stable)
            det_sigma_2d = torch.clamp(det_sigma_2d, min=1e-8)
        except RuntimeError:
            continue

        diff = pixel_coords - mean_2d.unsqueeze(0)
        mahalanobis_dist_sq = torch.sum(torch.matmul(diff, sigma_2d_inv) * diff, dim=1)

        gaussian_pdf_values_2d = (1.0 / (2 * torch.pi * torch.sqrt(det_sigma_2d))) * torch.exp(-0.5 * mahalanobis_dist_sq)
        alpha_per_pixel = (opacity_gaussian.squeeze() * gaussian_pdf_values_2d).unsqueeze(-1)
        alpha_per_pixel = torch.clamp(alpha_per_pixel, 0.0, 1.0)

        alpha_per_pixel_map = alpha_per_pixel.view(height, width, 1)
        color_gaussian_map = color.unsqueeze(0).unsqueeze(0).expand(height, width, -1)

        rendered_image = rendered_image + color_gaussian_map * alpha_per_pixel_map * alpha_accumulated_transmittance
        alpha_accumulated_transmittance = alpha_accumulated_transmittance * (1.0 - alpha_per_pixel_map)

    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    return rendered_image

