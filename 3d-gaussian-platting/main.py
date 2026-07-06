import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import initialize_gaussians,prune_gaussians,split_gaussians,generate_novel_viewpoints
from model import render_gaussian
from losses import combined_loss
from evaluate import calculate_lpips,calculate_psnr,calculate_ssim

# --- Main Execution Flow --- (using dummy data)

# 1. Initialize Gaussians
# Dummy data for initial point cloud
num_initial_points = 100
dummy_points_np = np.random.rand(num_initial_points, 3) * 10 - 5
dummy_colors_np = np.random.randint(0, 256, size=(num_initial_points, 3))

initial_gaussian_params = initialize_gaussians(dummy_points_np, dummy_colors_np)

# 2. Prepare Optimizable Parameters
gradient_gaussian_parameters = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for key, value in initial_gaussian_params.items():
    if isinstance(value, torch.Tensor):
        gradient_gaussian_parameters[key] = torch.nn.Parameter(value.clone().detach().to(device).requires_grad_(True))
    else:
        print(f"Warning: Skipping non-tensor parameter '{key}' from optimization.")

# 3. Initialize the Optimizer
learning_rate = 0.001
optimizer = optim.Adam(params=list(gradient_gaussian_parameters.values()), lr=learning_rate)

# 4. Structure the Optimization Loop with Adaptive Density Control
height, width = 100, 100
dummy_ground_truth_image = torch.rand(height, width, 3, dtype=torch.float32, device=device)
dummy_camera_intrinsics = {
    'focal_length': (torch.tensor(50.0), torch.tensor(50.0)),
    'principal_point': (torch.tensor(width / 2.0), torch.tensor(height / 2.0))
}
dummy_camera_extrinsics = {
    'rotation_matrix': np.eye(3),
    'translation_vector': np.array([0.0, 0.0, -5.0])
}

n_iterations = 100 # Number of training iterations for demonstration
pruning_interval = 20
splitting_interval = 40
initial_pruning_iterations = 10

print(f"\nStarting optimization loop for {n_iterations} iterations...")

for i in range(1, n_iterations + 1):
    optimizer.zero_grad()

    rendered_image = render_gaussian(
        means=gradient_gaussian_parameters['means'],
        scales=gradient_gaussian_parameters['scales'],
        rotations=gradient_gaussian_parameters['rotations'],
        opacities=gradient_gaussian_parameters['opacities'],
        sh_coeffs=gradient_gaussian_parameters['sh_coeffs'],
        camera_intrinsics=dummy_camera_intrinsics,
        camera_extrinsics=dummy_camera_extrinsics,
        image_dimensions=(height, width)
    )

    loss = combined_loss(rendered_image, dummy_ground_truth_image)
    loss.backward()
    optimizer.step()

    # Adaptive Density Control
    if i > initial_pruning_iterations and i % pruning_interval == 0:
        gradient_gaussian_parameters = prune_gaussians(gradient_gaussian_parameters)
        optimizer = optim.Adam(params=list(gradient_gaussian_parameters.values()), lr=learning_rate)

    if i > initial_pruning_iterations and i % splitting_interval == 0:
        gradient_gaussian_parameters = split_gaussians(gradient_gaussian_parameters)
        optimizer = optim.Adam(params=list(gradient_gaussian_parameters.values()), lr=learning_rate)

    if i % 10 == 0:
        print(f"Iteration [{i}/{n_iterations}], Loss: {loss.item():.4f}, Num Gaussians: {gradient_gaussian_parameters['means'].shape[0]}")

print("Optimization loop completed (with dummy data).")


# Generate novel viewpoints
base_intrinsics_novel = {
    'focal_length': (dummy_camera_intrinsics['focal_length'][0].item(), dummy_camera_intrinsics['focal_length'][1].item()),
    'principal_point': (dummy_camera_intrinsics['principal_point'][0].item(), dummy_camera_intrinsics['principal_point'][1].item())
}
base_extrinsics_novel = {
    'rotation_matrix': dummy_camera_extrinsics['rotation_matrix'],
    'translation_vector': dummy_camera_extrinsics['translation_vector']
}

num_novel_views = 5
novel_view_radius = 1.5
novel_view_elevation = 0.0
novel_view_center_point = np.array([0.0, 0.0, 0.0])

novel_camera_params_list = generate_novel_viewpoints(
    base_intrinsics=base_intrinsics_novel,
    base_extrinsics=base_extrinsics_novel,
    num_views=num_novel_views,
    radius=novel_view_radius,
    elevation=novel_view_elevation,
    center_point=novel_view_center_point
)

# Render from novel viewpoints
rendered_novel_views = []
print(f"\nRendering {num_novel_views} novel views...")
for i, cam_params in enumerate(novel_camera_params_list):
    with torch.no_grad():
        rendered_image_novel = render_gaussian(
            means=gradient_gaussian_parameters['means'],
            scales=gradient_gaussian_parameters['scales'],
            rotations=gradient_gaussian_parameters['rotations'],
            opacities=gradient_gaussian_parameters['opacities'],
            sh_coeffs=gradient_gaussian_parameters['sh_coeffs'],
            camera_intrinsics=cam_params['intrinsics'],
            camera_extrinsics=cam_params['extrinsics'],
            image_dimensions=(height, width)
        )
    rendered_novel_views.append(rendered_image_novel)
print(f"Successfully rendered {len(rendered_novel_views)} novel views.")

# Prepare ground truths for novel views (dummy)
novel_ground_truth_views = []
print(f"Generating dummy ground truth images for {len(rendered_novel_views)} novel views...")
for i in range(len(rendered_novel_views)):
    dummy_gt_image = torch.rand(height, width, 3, dtype=torch.float32, device=device)
    novel_ground_truth_views.append(dummy_gt_image)
print(f"Successfully generated {len(novel_ground_truth_views)} dummy ground truth views.")

# Report Metrics
psnr_values = []
ssim_values = []
lpips_values = []

print("\n--- Calculating Evaluation Metrics for Novel Views ---")
for i in range(len(rendered_novel_views)):
    rendered_img = rendered_novel_views[i]
    gt_img = novel_ground_truth_views[i]

    psnr = calculate_psnr(rendered_img, gt_img)
    psnr_values.append(psnr.item())

    ssim_val = calculate_ssim(rendered_img, gt_img)
    ssim_values.append(ssim_val.item())

    # LPIPS model is initialized inside calculate_lpips, which is not ideal for performance
    # For a real pipeline, initialize it once outside the loop.
    lpips_val = calculate_lpips(rendered_img, gt_img, device=device)
    lpips_values.append(lpips_val.item())

    print(f"View {i+1}: PSNR={psnr.item():.4f}, SSIM={ssim_val.item():.4f}, LPIPS={lpips_val.item():.4f}")

print("\n--- Average Metrics Across Novel Views ---")
print(f"Average PSNR: {np.mean(psnr_values):.4f}")
print(f"Average SSIM: {np.mean(ssim_values):.4f}")
print(f"Average LPIPS: {np.mean(lpips_values):.4f}")
print("------------------------------------------")

# Visualize Results
num_views_to_show = min(3, len(rendered_novel_views))

plt.figure(figsize=(15, num_views_to_show * 5))
for i in range(num_views_to_show):
    rendered_img_np = rendered_novel_views[i].cpu().numpy()
    gt_img_np = novel_ground_truth_views[i].cpu().numpy()

    plt.subplot(num_views_to_show, 2, 2 * i + 1)
    plt.imshow(rendered_img_np)
    plt.title(f"Rendered View {i+1}")
    plt.axis('off')

    plt.subplot(num_views_to_show, 2, 2 * i + 2)
    plt.imshow(gt_img_np)
    plt.title(f"Ground Truth View {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()