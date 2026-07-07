import torch
import math

def generate_anchor_boxes(img_size, feature_map_sizes, scales, aspect_ratios, device='cpu'):
    """
    Generates anchor boxes for all feature maps.

    Args:
        img_size (int): Input image size (e.g., 300 for 300x300).
        feature_map_sizes (list of tuples): List of (height, width) for each feature map.
        scales (list of float): List of base scales for each feature map.
        aspect_ratios (list of list of float): List of aspect ratios for each feature map.
                                             Each inner list corresponds to a feature map.
        device (str): Device to place tensors on.

    Returns:
        torch.Tensor: A tensor of anchor boxes in (cx, cy, w, h) format, normalized [0, 1].
                      Shape: (total_num_anchors, 4)
    """
    all_anchors = []

    # Ensure the number of scales matches the number of feature maps
    if len(scales) != len(feature_map_sizes):
        raise ValueError("Number of scales must match number of feature maps.")
    if len(aspect_ratios) != len(feature_map_sizes):
        raise ValueError("Number of aspect ratios lists must match number of feature maps.")

    for k, (fm_h, fm_w) in enumerate(feature_map_sizes):
        for i in range(fm_h):
            for j in range(fm_w):
                # Center of the anchor box relative to the feature map cell
                # Normalized to [0, 1] range of the input image
                cx = (j + 0.5) / fm_w
                cy = (i + 0.5) / fm_h

                # Current feature map's base scale
                s_k = scales[k]

                # 1. Anchor with aspect ratio 1:1 and scale s_k
                w = h = s_k
                all_anchors.append([cx, cy, w, h])

                # 2. Anchor with aspect ratio 1:1 and an additional scale
                # For the last feature map, use the same scale for s_k+1 or a predefined max_scale
                if k == len(scales) - 1: # Last feature map
                    s_k_plus_1 = 1.0 # Common practice or scales[k] itself
                else:
                    s_k_plus_1 = scales[k+1]

                s_k_prime = math.sqrt(s_k * s_k_plus_1)
                w = h = s_k_prime
                all_anchors.append([cx, cy, w, h])

                # 3. Anchors with other aspect ratios
                for ar in aspect_ratios[k]:
                    if ar == 1: # Skip 1:1 as it's already handled
                        continue

                    # Width and height calculation based on aspect ratio
                    w = s_k * math.sqrt(ar)
                    h = s_k / math.sqrt(ar)
                    all_anchors.append([cx, cy, w, h])

                    # Also add the inverse aspect ratio
                    w = s_k / math.sqrt(ar)
                    h = s_k * math.sqrt(ar)
                    all_anchors.append([cx, cy, w, h])

    # Convert list of anchors to a tensor
    anchors_cxcywh = torch.tensor(all_anchors, dtype=torch.float32, device=device)

    # Convert (cx, cy, w, h) to (xmin, ymin, xmax, ymax) format
    anchors_xyxy = torch.empty_like(anchors_cxcywh)
    anchors_xyxy[:, 0] = anchors_cxcywh[:, 0] - anchors_cxcywh[:, 2] / 2 # xmin
    anchors_xyxy[:, 1] = anchors_cxcywh[:, 1] - anchors_cxcywh[:, 3] / 2 # ymin
    anchors_xyxy[:, 2] = anchors_cxcywh[:, 0] + anchors_cxcywh[:, 2] / 2 # xmax
    anchors_xyxy[:, 3] = anchors_cxcywh[:, 1] + anchors_cxcywh[:, 3] / 2 # ymax

    # Clip any anchor boxes that go beyond the [0, 1] boundaries (though they shouldn't with correct params)
    anchors_xyxy = torch.clamp(anchors_xyxy, 0.0, 1.0)

    return anchors_xyxy

def box_to_center_form(boxes):
    """
    Convert boxes from (xmin, ymin, xmax, ymax) to (cx, cy, w, h).
    Args:
        boxes (tensor): A tensor of boxes in (xmin, ymin, xmax, ymax) format. Size: (N, 4)
    Returns:
        tensor: A tensor of boxes in (cx, cy, w, h) format. Size: (N, 4)
    """
    return torch.cat([
        ((boxes[:, 0] + boxes[:, 2]) / 2).unsqueeze(1),  # cx
        ((boxes[:, 1] + boxes[:, 3]) / 2).unsqueeze(1),  # cy
        (boxes[:, 2] - boxes[:, 0]).unsqueeze(1),      # w
        (boxes[:, 3] - boxes[:, 1]).unsqueeze(1)       # h
    ], 1)

def center_to_corner_form(boxes):
    """
    Convert boxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax).
    Args:
        boxes (tensor): A tensor of boxes in (cx, cy, w, h) format. Size: (N, 4)
    Returns:
        tensor: A tensor of boxes in (xmin, ymin, xmax, ymax) format. Size: (N, 4)
    """
    return torch.cat([
        (boxes[:, 0] - boxes[:, 2] / 2).unsqueeze(1),  # xmin
        (boxes[:, 1] - boxes[:, 3] / 2).unsqueeze(1),  # ymin
        (boxes[:, 0] + boxes[:, 2] / 2).unsqueeze(1),  # xmax
        (boxes[:, 1] + boxes[:, 3] / 2).unsqueeze(1)   # ymax
    ], 1)

def iou_of(boxes_a, boxes_b):
    """
    Calculates the Intersection over Union (IoU) between two sets of bounding boxes.
    Args:
        boxes_a (tensor): Ground truth boxes. Shape: (N, 4)
        boxes_b (tensor): Anchor boxes. Shape: (M, 4)
    Returns:
        tensor: IoU matrix. Shape: (N, M)
    """
    # Ensure boxes are in (xmin, ymin, xmax, ymax) format

    N = boxes_a.size(0)
    M = boxes_b.size(0)

    # Calculate intersection coordinates
    inter_xmin = torch.max(boxes_a[:, 0].unsqueeze(1), boxes_b[:, 0].unsqueeze(0))
    inter_ymin = torch.max(boxes_a[:, 1].unsqueeze(1), boxes_b[:, 1].unsqueeze(0))
    inter_xmax = torch.min(boxes_a[:, 2].unsqueeze(1), boxes_b[:, 2].unsqueeze(0))
    inter_ymax = torch.min(boxes_a[:, 3].unsqueeze(1), boxes_b[:, 3].unsqueeze(0))

    # Calculate intersection area
    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_w * inter_h

    # Calculate area of boxes_a and boxes_b
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # Calculate union area
    union_area = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter_area

    # Calculate IoU
    iou = inter_area / union_area

    return iou

def assign_targets_to_anchors(
    ground_truth_boxes, ground_truth_labels, anchors, num_classes,
    iou_threshold=0.5, neg_pos_ratio=3, device='cpu', variances=(0.1, 0.1, 0.2, 0.2)
):
    """
    Assigns ground truth boxes and labels to anchor boxes.

    Args:
        ground_truth_boxes (tensor): Ground truth bounding boxes. Shape: (num_gt, 4) (xmin, ymin, xmax, ymax)
        ground_truth_labels (tensor): Ground truth class labels. Shape: (num_gt)
        anchors (tensor): Generated anchor boxes. Shape: (num_anchors, 4) (xmin, ymin, xmax, ymax)
        num_classes (int): Total number of classes (including background).
        iou_threshold (float): IoU threshold for considering an anchor as positive.
        neg_pos_ratio (int): Ratio of negative to positive samples for hard negative mining.
        device (str): Device to place tensors on.
        variances (tuple): Variances used for encoding regression targets.

    Returns:
        tuple: (encoded_boxes, assigned_labels)
            encoded_boxes (tensor): Regression targets for anchors. Shape: (num_anchors, 4)
            assigned_labels (tensor): Class labels for anchors. Shape: (num_anchors)
    """
    num_anchors = anchors.size(0)
    num_gt = ground_truth_boxes.size(0)

    # Initialize tensors
    # (num_anchors) stores the index of the best gt for each anchor
    best_gt_for_anchor = torch.zeros(num_anchors, dtype=torch.long, device=device)
    # (num_anchors) stores the max IoU for each anchor with any gt
    best_iou_for_anchor = torch.zeros(num_anchors, dtype=torch.float32, device=device)
    # (num_gt) stores the index of the best anchor for each gt
    best_anchor_for_gt = torch.zeros(num_gt, dtype=torch.long, device=device)
    # (num_gt) stores the max IoU for each gt with any anchor
    best_iou_for_gt = torch.zeros(num_gt, dtype=torch.float32, device=device)

    if num_gt > 0: # Only proceed if there are ground truth boxes
        iou_matrix = iou_of(ground_truth_boxes, anchors) # (num_gt, num_anchors)

        # Find best IoU for each anchor with a ground truth box
        best_iou_for_anchor, best_gt_for_anchor = iou_matrix.max(dim=0) # (num_anchors)

        # Find best IoU for each ground truth box with an anchor
        best_iou_for_gt, best_anchor_for_gt = iou_matrix.max(dim=1) # (num_gt)

        # Ensure each ground truth box is matched with its best anchor (highest IoU)
        # This handles cases where the best anchor for a GT box has low IoU and might be missed by thresholding
        for gt_idx in range(num_gt):
            best_gt_for_anchor[best_anchor_for_gt[gt_idx]] = gt_idx
            best_iou_for_anchor[best_anchor_for_gt[gt_idx]] = 1.0 # Force IoU to 1.0 for the best match

    # Initialize assigned labels and boxes for all anchors
    # Default to background class (0)
    assigned_labels = torch.zeros(num_anchors, dtype=torch.long, device=device) # Background class is 0
    # Default assigned boxes are just anchors themselves (will be updated for positives)
    assigned_boxes = anchors.clone()
    # Initialize encoded_boxes as well
    encoded_boxes = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

    # Assign labels and boxes for positive anchors
    # Anchors with IoU > threshold are positive
    positive_mask = best_iou_for_anchor >= iou_threshold
    if positive_mask.any(): # Only assign if there are positive matches
        # Assign ground truth labels to positive anchors
        assigned_labels[positive_mask] = ground_truth_labels[best_gt_for_anchor[positive_mask]]
        # Assign ground truth boxes to positive anchors
        # This is needed to calculate the offsets (regression targets)
        assigned_boxes[positive_mask] = ground_truth_boxes[best_gt_for_anchor[positive_mask]]

        pos_anchors_cxcywh = box_to_center_form(anchors[positive_mask])
        pos_assigned_boxes_cxcywh = box_to_center_form(assigned_boxes[positive_mask])

        # SSD Encoding scheme
        encoded_boxes[positive_mask, 0] = (pos_assigned_boxes_cxcywh[:, 0] - pos_anchors_cxcywh[:, 0]) / pos_anchors_cxcywh[:, 2] / variances[0]
        encoded_boxes[positive_mask, 1] = (pos_assigned_boxes_cxcywh[:, 1] - pos_anchors_cxcywh[:, 1]) / pos_anchors_cxcywh[:, 3] / variances[1]
        encoded_boxes[positive_mask, 2] = torch.log(pos_assigned_boxes_cxcywh[:, 2] / pos_anchors_cxcywh[:, 2]) / variances[2]
        encoded_boxes[positive_mask, 3] = torch.log(pos_assigned_boxes_cxcywh[:, 3] / pos_anchors_cxcywh[:, 3]) / variances[3]

    return encoded_boxes, assigned_labels, positive_mask

def decode_boxes(rel_codes, anchors_xyxy, variances=(0.1, 0.1, 0.2, 0.2)):
    """
    Converts predicted relative box codes to absolute bounding box coordinates.
    Args:
        rel_codes (tensor): Predicted regression targets (dx, dy, dw, dh). Shape: (N, 4)
        anchors_xyxy (tensor): Anchor boxes in (xmin, ymin, xmax, ymax) format. Shape: (N, 4)
        variances (tuple): Variances used in encoding.
    Returns:
        tensor: Decoded bounding boxes in (xmin, ymin, xmax, ymax) format. Shape: (N, 4)
    """
    # Convert anchors to cx, cy, w, h format
    anchors_cxcywh = box_to_center_form(anchors_xyxy)

    # Unpack variances
    variance_x, variance_y, variance_w, variance_h = variances

    # Decode center coordinates
    pred_cx = rel_codes[:, 0] * variance_x * anchors_cxcywh[:, 2] + anchors_cxcywh[:, 0]
    pred_cy = rel_codes[:, 1] * variance_y * anchors_cxcywh[:, 3] + anchors_cxcywh[:, 1]

    # Decode width and height
    pred_w = torch.exp(rel_codes[:, 2] * variance_w) * anchors_cxcywh[:, 2]
    pred_h = torch.exp(rel_codes[:, 3] * variance_h) * anchors_cxcywh[:, 3]

    # Convert decoded cx, cy, w, h back to xmin, ymin, xmax, ymax
    decoded_boxes = center_to_corner_form(torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1))

    # Clamp coordinates to [0, 1] range
    decoded_boxes = torch.clamp(decoded_boxes, 0.0, 1.0)
    return decoded_boxes