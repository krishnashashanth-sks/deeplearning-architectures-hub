import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

def combined_loss(rendered_image: torch.Tensor, ground_truth_image: torch.Tensor, lambda_l1: float = 0.8, lambda_ssim: float = 0.2) -> torch.Tensor:
    """
    Calculates a combined L1 and (1 - SSIM) loss.
    """
    l1_loss = F.l1_loss(rendered_image, ground_truth_image)

    rendered_image_nchw = rendered_image.permute(2, 0, 1).unsqueeze(0)
    ground_truth_image_nchw = ground_truth_image.permute(2, 0, 1).unsqueeze(0)

    ssim_val = ssim(rendered_image_nchw, ground_truth_image_nchw, data_range=1.0, size_average=True)
    ssim_loss = 1.0 - ssim_val

    total_loss = lambda_l1 * l1_loss + lambda_ssim * ssim_loss

    return total_loss

