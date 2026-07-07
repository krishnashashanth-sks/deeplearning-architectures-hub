from utils import decode_boxes
import torchvision.ops as ops
import torch.nn.functional as F
import torch

def nms(boxes, scores, iou_threshold):
    """
    Performs Non-Maximum Suppression (NMS) on the bounding boxes.

    Args:
        boxes (tensor): Bounding box coordinates in (xmin, ymin, xmax, ymax) format. Shape: (N, 4)
        scores (tensor): Confidence scores for each box. Shape: (N)
        iou_threshold (float): IoU threshold for NMS.

    Returns:
        tensor: Indices of the boxes to keep after NMS.
    """
    return ops.nms(boxes, scores, iou_threshold)

def post_process_detections(pred_cls, pred_reg, anchors, num_classes,
                          conf_threshold=0.05, nms_iou_threshold=0.45, top_k=200, variances=(0.1, 0.1, 0.2, 0.2), device='cpu'):
    """
    Performs post-processing to get final object detections.

    Args:
        pred_cls (tensor): Predicted classification scores. Shape: (num_anchors, num_classes)
        pred_reg (tensor): Predicted regression offsets. Shape: (num_anchors, 4)
        anchors (tensor): Anchor boxes in (xmin, ymin, xmax, ymax) format. Shape: (num_anchors, 4)
        num_classes (int): Total number of classes (including background).
        conf_threshold (float): Confidence threshold for filtering initial predictions.
        nms_iou_threshold (float): IoU threshold for NMS.
        top_k (int): Number of top detections to keep after NMS.
        variances (tuple): Variances used in encoding/decoding.
        device (str): Device for tensor operations.

    Returns:
        tuple: (final_boxes, final_labels, final_scores)
            final_boxes (tensor): Decoded and NMS-filtered bounding boxes. Shape: (num_detections, 4)
            final_labels (tensor): Class labels for final detections. Shape: (num_detections)
            final_scores (tensor): Confidence scores for final detections. Shape: (num_detections)
    """
    # Move tensors to CPU for NMS if they are on GPU, as torchvision.ops.nms might not support it directly
    # (or ensure torchvision version supports GPU NMS)
    pred_cls = pred_cls.to(device)
    pred_reg = pred_reg.to(device)
    anchors = anchors.to(device)

    decoded_boxes = decode_boxes(pred_reg, anchors, variances=variances)

    final_boxes_list = []
    final_scores_list = []
    final_labels_list = []

    for c in range(1, num_classes): # Iterate over classes, excluding background (class 0)
        class_scores = F.softmax(pred_cls[:, c], dim=0) # Get scores for current class

        # Filter by confidence threshold
        mask = class_scores > conf_threshold
        if mask.sum() == 0: # No detections for this class
            continue

        boxes_filtered = decoded_boxes[mask]
        scores_filtered = class_scores[mask]

        # Apply NMS
        keep_indices = nms(boxes_filtered, scores_filtered, nms_iou_threshold)

        # Keep top_k detections
        if len(keep_indices) > top_k:
            scores_sorted, score_sort_indices = scores_filtered[keep_indices].sort(descending=True)
            keep_indices = keep_indices[score_sort_indices[:top_k]]

        final_boxes_list.append(boxes_filtered[keep_indices])
        final_scores_list.append(scores_filtered[keep_indices])
        final_labels_list.append(torch.full_like(scores_filtered[keep_indices], fill_value=c, dtype=torch.long))

    if len(final_boxes_list) == 0:
        return torch.empty(0, 4, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, device=device)

    final_boxes = torch.cat(final_boxes_list, dim=0)
    final_scores = torch.cat(final_scores_list, dim=0)
    final_labels = torch.cat(final_labels_list, dim=0)

    # Sort all detections by score and keep top_k overall
    if len(final_scores) > top_k:
        scores_sorted, score_sort_indices = final_scores.sort(descending=True)
        final_boxes = final_boxes[score_sort_indices[:top_k]]
        final_scores = final_scores[score_sort_indices[:top_k]]
        final_labels = final_labels[score_sort_indices[:top_k]]

    return final_boxes, final_labels, final_scores