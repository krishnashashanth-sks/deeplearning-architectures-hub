import torch
import torchvision.ops as ops

def decode_predictions(predictions, anchors, img_size, num_classes):
    batch_size = predictions.shape[0]
    num_anchors = predictions.shape[1]
    grid_h, grid_w = predictions.shape[2:4]

    pred_bbox_raw = predictions[..., :4]
    pred_conf_logits = predictions[..., 4]
    pred_cls_logits = predictions[..., 5:]

    pred_conf = torch.sigmoid(pred_conf_logits)
    pred_cls = torch.sigmoid(pred_cls_logits)

    grid_x = torch.arange(grid_w, device=predictions.device).repeat(grid_h, 1).view([1, 1, grid_h, grid_w]).float()
    grid_y = torch.arange(grid_h, device=predictions.device).repeat(grid_w, 1).t().view([1, 1, grid_h, grid_w]).float()

    anchors_tensor = anchors.view(1, num_anchors, 1, 1, 2).to(predictions.device)

    bx = torch.sigmoid(pred_bbox_raw[..., 0]) + grid_x
    by = torch.sigmoid(pred_bbox_raw[..., 1]) + grid_y
    bw = torch.exp(pred_bbox_raw[..., 2]) * anchors_tensor[..., 0] * grid_w
    bh = torch.exp(pred_bbox_raw[..., 3]) * anchors_tensor[..., 1] * grid_h

    bx /= grid_w
    by /= grid_h
    bw /= grid_w
    bh /= grid_h

    decoded_boxes = torch.stack((bx, by, bw, bh), dim=-1).view(batch_size, -1, 4)
    decoded_conf = pred_conf.view(batch_size, -1, 1)
    decoded_cls = pred_cls.view(batch_size, -1, num_classes)

    detections = torch.cat((decoded_boxes, decoded_conf, decoded_cls), dim=-1)

    return detections

def apply_confidence_threshold(detections, conf_threshold):
    conf_mask = (detections[..., 4] >= conf_threshold).unsqueeze(-1)
    filtered_detections = detections * conf_mask.float()
    return filtered_detections

def apply_nms(detections, iou_threshold, conf_threshold=0.0):
    batch_size = detections.shape[0]
    final_detections = []

    for i in range(batch_size):
        image_detections = detections[i]

        valid_mask = image_detections[:, 4] > conf_threshold
        image_detections = image_detections[valid_mask]

        if image_detections.numel() == 0:
            final_detections.append(torch.empty((0, detections.shape[-1]), device=detections.device))
            continue

        boxes_xywh = image_detections[:, :4]
        boxes_x1y1x2y2 = torch.zeros_like(boxes_xywh)
        boxes_x1y1x2y2[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_x1y1x2y2[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_x1y1x2y2[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_x1y1x2y2[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        scores = image_detections[:, 4]
        class_scores = image_detections[:, 5:]

        max_class_scores, class_preds = torch.max(class_scores, dim=1)
        overall_scores = scores * max_class_scores

        overall_valid_mask = overall_scores > conf_threshold
        boxes_x1y1x2y2 = boxes_x1y1x2y2[overall_valid_mask]
        overall_scores = overall_scores[overall_valid_mask]
        image_detections = image_detections[overall_valid_mask]

        if image_detections.numel() == 0:
            final_detections.append(torch.empty((0, detections.shape[-1]), device=detections.device))
            continue

        keep_indices = ops.nms(boxes_x1y1x2y2, overall_scores, iou_threshold)
        kept_detections = image_detections[keep_indices]

        final_detections.append(kept_detections)

    return final_detections
