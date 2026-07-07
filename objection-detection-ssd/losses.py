import torch
import torch.nn as nn
import torch.nn.functional as F

class SSDLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3, alpha=1.0, variances=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha  # Weight for localization loss
        self.variances = variances
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred_cls, pred_reg, targets_cls, targets_reg):
        """
        Args:
            pred_cls (tensor): Predicted classification scores. Shape: (N, num_anchors, num_classes)
            pred_reg (tensor): Predicted regression offsets. Shape: (N, num_anchors, 4)
            targets_cls (tensor): Ground truth class labels for anchors. Shape: (N, num_anchors)
            targets_reg (tensor): Ground truth regression targets for anchors. Shape: (N, num_anchors, 4)
        """
        batch_size = pred_cls.size(0)
        num_anchors = pred_cls.size(1)
        num_classes = pred_cls.size(2)

        # Reshape predictions for easier processing
        pred_cls = pred_cls.view(-1, num_classes) # (N * num_anchors, num_classes)
        pred_reg = pred_reg.view(-1, 4)           # (N * num_anchors, 4)
        targets_cls = targets_cls.view(-1)         # (N * num_anchors)
        targets_reg = targets_reg.view(-1, 4)      # (N * num_anchors, 4)

        # Identify positive anchors (non-background class)
        pos_mask = targets_cls > 0
        num_pos = pos_mask.long().sum(dim=0)

        # 1. Localization Loss (Lloc)
        # Only calculate for positive anchors
        if num_pos > 0:
            pos_pred_reg = pred_reg[pos_mask]
            pos_targets_reg = targets_reg[pos_mask]
            loc_loss = self.smooth_l1_loss(pos_pred_reg, pos_targets_reg)
        else:
            loc_loss = torch.tensor(0.0, device=pred_cls.device)

        # 2. Confidence Loss (Lconf)
        # Calculate cross-entropy loss for all anchors
        # For hard negative mining, we need per-anchor loss
        cls_loss_all = F.cross_entropy(pred_cls, targets_cls, reduction='none') # (N * num_anchors)

        # Hard Negative Mining
        # Don't consider positive anchors as negatives
        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[pos_mask] = 0.0 # Zero out loss for positive anchors

        # Reshape to (batch_size, num_anchors) for per-image mining
        cls_loss_neg = cls_loss_neg.view(batch_size, -1)

        # Sort negative losses in descending order
        _, neg_idx = cls_loss_neg.sort(dim=1, descending=True)
        _, rank_idx = neg_idx.sort(dim=1) # Get rank of each anchor

        # Select top `neg_pos_ratio * num_pos_per_image` negatives per image
        # Note: num_pos could be 0 for some images
        num_pos_per_image = pos_mask.view(batch_size, -1).long().sum(dim=1)
        num_neg_per_image = torch.clamp(self.neg_pos_ratio * num_pos_per_image, min=1)

        # Create a mask for selected negative anchors
        neg_mask_per_image = rank_idx < num_neg_per_image.unsqueeze(1)

        # Combine positive and selected negative masks
        pos_neg_mask = pos_mask.view(batch_size, -1) | neg_mask_per_image # (batch_size, num_anchors)
        pos_neg_mask = pos_neg_mask.view(-1) # Flatten back to (N * num_anchors)

        # Final confidence loss over positive and hard negative samples
        conf_loss = F.cross_entropy(pred_cls[pos_neg_mask], targets_cls[pos_neg_mask], reduction='sum')

        # Total loss
        # Divide by num_pos to normalize by the number of positive samples,
        # which is a common practice in SSD to balance the loss contributions.
        N = num_pos.item() if num_pos > 0 else 1 # Avoid division by zero
        total_loss = (self.alpha * loc_loss + conf_loss) / N

        return total_loss, loc_loss / N, conf_loss / N