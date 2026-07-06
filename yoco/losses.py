import torch
import torch.nn.functional as F
import torch.nn as nn

# ---  Custom Loss Functions ---

# Helper 1: Convert bounding box parameters (tx, ty, tw, th) to absolute (x, y, w, h)
def box_parameters_to_coords(p_bbox, anchors, grid_size):
    tx = p_bbox[..., 0]
    ty = p_bbox[..., 1]
    tw = p_bbox[..., 2]
    th = p_bbox[..., 3]

    grid_x = torch.arange(grid_size, device=p_bbox.device).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).float()
    grid_y = torch.arange(grid_size, device=p_bbox.device).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).float()

    anchor_w = anchors[:, 0].view(1, -1, 1, 1)
    anchor_h = anchors[:, 1].view(1, -1, 1, 1)

    bx = torch.sigmoid(tx) + grid_x
    by = torch.sigmoid(ty) + grid_y

    bw = torch.exp(tw) * anchor_w
    bh = torch.exp(th) * anchor_h

    bx /= grid_size
    by /= grid_size
    bw /= grid_size
    bh /= grid_size

    return torch.stack((bx, by, bw, bh), dim=-1)

# Helper 2: Calculate Intersection over Union (IoU) of two bounding boxes
def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = (box1[..., 0] - box1[..., 2] / 2,
                                  box1[..., 1] - box1[..., 3] / 2,
                                  box1[..., 0] + box1[..., 2] / 2,
                                  box1[..., 1] + box1[..., 3] / 2)
    b2_x1, b2_y1, b2_x2, b2_y2 = (box2[..., 0] - box2[..., 2] / 2,
                                  box2[..., 1] - box2[..., 3] / 2,
                                  box2[..., 0] + box2[..., 2] / 2,
                                  box2[..., 1] + box2[..., 3] / 2)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area + b2_area - inter_area

    iou = inter_area / (union_area + 1e-16)

    return iou

# Helper 3: Map ground truth boxes to grid cells and anchor boxes
def build_targets(pred_boxes_raw, target, anchors, num_anchors, num_classes, grid_size, ignore_threshold=0.5):
    batch_size = target.shape[0]

    obj_mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size, dtype=torch.bool, device=target.device)
    noobj_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size, dtype=torch.bool, device=target.device)
    bbox_targets = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 4, device=target.device)
    class_targets = torch.zeros(batch_size, num_anchors, grid_size, grid_size, num_classes, device=target.device)

    anchors_grid = anchors.clone().to(target.device)

    for b in range(batch_size):
        # Get predicted boxes for this batch item, converted to (x,y,w,h) for ignore_threshold calculation
        all_pred_boxes_b = box_parameters_to_coords(pred_boxes_raw[b].unsqueeze(0), anchors, grid_size).squeeze(0)
        all_pred_boxes_flat = all_pred_boxes_b.view(-1, 4) # (num_total_predictions, 4)

        for t in range(target.shape[1]):
            gt_class_id = int(target[b, t, 0])
            if gt_class_id == -1:
                continue

            gt_x = target[b, t, 1]
            gt_y = target[b, t, 2]
            gt_w = target[b, t, 3]
            gt_h = target[b, t, 4]

            # Convert GT to grid scale for cell assignment
            gt_x_grid = gt_x * grid_size
            gt_y_grid = gt_y * grid_size
            gt_w_grid = gt_w * grid_size
            gt_h_grid = gt_h * grid_size

            gi = int(gt_x_grid)
            gj = int(gt_y_grid)

            gt_box_for_anchor_match = torch.tensor([0, 0, gt_w_grid, gt_h_grid], device=target.device).unsqueeze(0)

            anchor_boxes = torch.cat((torch.zeros_like(anchors_grid), anchors_grid * grid_size), 1)
            iou_anchors = bbox_iou(gt_box_for_anchor_match, anchor_boxes)

            best_iou, best_anchor_idx = iou_anchors.max(0)

            obj_mask[b, best_anchor_idx, gj, gi] = True
            noobj_mask[b, best_anchor_idx, gj, gi] = False

            target_x = gt_x_grid - gi
            target_y = gt_y_grid - gj
            target_w = torch.log(gt_w_grid / (anchors_grid[best_anchor_idx, 0] * grid_size) + 1e-16)
            target_h = torch.log(gt_h_grid / (anchors_grid[best_anchor_idx, 1] * grid_size) + 1e-16)

            bbox_targets[b, best_anchor_idx, gj, gi, 0] = target_x
            bbox_targets[b, best_anchor_idx, gj, gi, 1] = target_y
            bbox_targets[b, best_anchor_idx, gj, gi, 2] = target_w
            bbox_targets[b, best_anchor_idx, gj, gi, 3] = target_h

            class_targets[b, best_anchor_idx, gj, gi, gt_class_id] = 1

            # Ignore predictions with high IoU to GT but not the best anchor
            gt_box_normalized = target[b, t, 1:]
            ious_with_gt = bbox_iou(gt_box_normalized.unsqueeze(0), all_pred_boxes_flat)
            ious_reshaped = ious_with_gt.view(num_anchors, grid_size, grid_size)
            noobj_mask[b][ious_reshaped > ignore_threshold] = False

    return obj_mask, noobj_mask, bbox_targets, class_targets

def compute_localization_loss(pred_bbox_params, target_bbox_params, obj_mask):
    masked_pred_bbox = pred_bbox_params[obj_mask]
    masked_target_bbox = target_bbox_params[obj_mask]

    if masked_pred_bbox.numel() == 0:
        return torch.tensor(0.0, device=pred_bbox_params.device)

    localization_loss = F.smooth_l1_loss(masked_pred_bbox, masked_target_bbox, reduction='sum')
    return localization_loss

# Helper 5: Confidence Loss Component
def compute_confidence_loss(pred_conf, obj_mask, noobj_mask, bce_loss_fn):
    target_conf = torch.zeros_like(pred_conf, dtype=torch.float, device=pred_conf.device)
    target_conf[obj_mask] = 1.0

    loss_obj = bce_loss_fn(pred_conf[obj_mask], target_conf[obj_mask])
    loss_noobj = bce_loss_fn(pred_conf[noobj_mask], target_conf[noobj_mask])

    confidence_loss = loss_obj + loss_noobj
    return confidence_loss

# Helper 6: Classification Loss Component
def compute_classification_loss(pred_cls, target_cls, obj_mask):
    masked_pred_cls = pred_cls[obj_mask]
    masked_target_cls = target_cls[obj_mask]

    if masked_pred_cls.numel() == 0:
        return torch.tensor(0.0, device=pred_cls.device)

    classification_loss = F.binary_cross_entropy_with_logits(
        masked_pred_cls, masked_target_cls.float(), reduction='sum'
    )
    return classification_loss

class Yocoloss(nn.Module):
    def __init__(self, anchors, num_classes, img_size=416, ignore_threshold=0.5,
                 lambda_coord=5.0, lambda_noobj=0.5):
        super(Yocoloss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.ignore_threshold = ignore_threshold
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]
        grid_size = predictions.shape[2]

        pred_bbox_raw = predictions[..., :4]
        pred_conf = predictions[..., 4]
        pred_cls = predictions[..., 5:]

        obj_mask, noobj_mask, bbox_targets, class_targets = build_targets(
            pred_bbox_raw.detach(), targets, self.anchors, self.num_anchors,
            self.num_classes, grid_size, self.ignore_threshold
        )

        obj_mask = obj_mask.to(predictions.device)
        noobj_mask = noobj_mask.to(predictions.device)
        bbox_targets = bbox_targets.to(predictions.device)
        class_targets = class_targets.to(predictions.device)

        pred_bbox_raw = pred_bbox_raw.view(batch_size, self.num_anchors, grid_size, grid_size, 4)
        pred_conf = pred_conf.view(batch_size, self.num_anchors, grid_size, grid_size)
        pred_cls = pred_cls.view(batch_size, self.num_anchors, grid_size, grid_size, self.num_classes)

        loss_bbox = compute_localization_loss(
            pred_bbox_raw, bbox_targets, obj_mask
        )

        loss_conf = compute_confidence_loss(pred_conf, obj_mask, noobj_mask, self.bce_loss)

        loss_cls = compute_classification_loss(
            pred_cls, class_targets, obj_mask
        )

        total_loss = self.lambda_coord * loss_bbox + loss_conf + loss_cls

        return total_loss
