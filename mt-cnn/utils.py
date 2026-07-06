import numpy as np
import cv2

# ---  Utility Functions ---

def load_image(image_path):
    """
    Conceptual function to load an image. In a real scenario, this would use cv2.imread or similar.
    Returns a dummy image array for demonstration.
    """
    # For this consolidated inference, we return a fixed dummy image
    # if it's called during a test run, or actual image if called by mtcnn_detect_faces
    print(f"Simulating loading image: {image_path}")
    return np.random.randint(0, 255, size=(600, 800, 3), dtype=np.uint8)

def iou(box, boxes):
    """
    Calculate IoU (Intersection over Union) of a box with respect to a list of boxes.
    box: [x1, y1, x2, y2]
    boxes: Nx4 array of [x1, y1, x2, y2] ground truth boxes
    """
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # obtain coordinates of the intersection box
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the intersection box
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)

    # compute the area of intersection
    inter = w * h

    # compute the area of union
    union = box_area + areas - inter

    iou_values = inter / (union + 1e-6) # Add epsilon to avoid division by zero
    return iou_values

def bbox_regression_target(anchor_boxes, gt_boxes):
    """
    Calculate bounding box regression targets (dx, dy, dw, dh).
    anchor_boxes: Nx4 array of [x1, y1, x2, y2] (proposal boxes)
    gt_boxes: Nx4 array of [x1, y1, x2, y2] (ground truth boxes)
    """
    anchor_width = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    anchor_height = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    anchor_center_x = anchor_boxes[:, 0] + 0.5 * anchor_width
    anchor_center_y = anchor_boxes[:, 1] + 0.5 * anchor_height

    gt_width = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_height = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_x = gt_boxes[:, 0] + 0.5 * gt_width
    gt_center_y = gt_boxes[:, 1] + 0.5 * gt_height

    # Calculate regression targets (t_x, t_y, t_w, t_h)
    targets_dx = (gt_center_x - anchor_center_x) / anchor_width
    targets_dy = (gt_center_y - anchor_center_y) / anchor_height
    targets_dw = np.log(gt_width / anchor_width)
    targets_dh = np.log(gt_height / anchor_height)

    return np.vstack([targets_dx, targets_dy, targets_dw, targets_dh]).transpose()

def nms(boxes, scores, threshold, method='union'):
    """
    Apply Non-Maximum Suppression.
    boxes: numpy array of shape (N, 4) [x1, y1, x2, y2]
    scores: numpy array of shape (N,)
    threshold: IoU threshold
    method: 'union' or 'min' for IoU calculation
    """
    if boxes.shape[0] == 0:
        return np.array([])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if method == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif method == 'min':
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            raise ValueError("Unknown method: {}".format(method))

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)

def generate_image_pyramid(image, min_size=12, scale_factor=0.709):
    """
    Generates an image pyramid from the input image.
    image: The original input image (numpy array).
    min_size: The minimum dimension (width or height) for the smallest image in the pyramid.
    scale_factor: The factor by which to scale down the image at each step.
    Returns: A list of scaled images.
    """
    scales = []
    current_image = image.copy()
    h, w = current_image.shape[:2]

    min_side = min(h, w)

    while min_side >= min_size:
        scales.append(current_image)
        w_new = int(w * scale_factor)
        h_new = int(h * scale_factor)

        if w_new < min_size or h_new < min_size:
            break

        current_image = cv2.resize(current_image, (w_new, h_new), interpolation=cv2.INTER_AREA)
        h, w = current_image.shape[:2]
        min_side = min(h, w)

    return scales

def adjust_boxes(bboxes, reg_targets):
    """
    Adjust bounding boxes using regression targets.
    bboxes: Nx4 array of [x1, y1, x2, y2]
    reg_targets: Nx4 array of [dx, dy, dw, dh]
    """
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    x1 = bboxes[:, 0] + reg_targets[:, 0] * w
    y1 = bboxes[:, 1] + reg_targets[:, 1] * h
    x2 = bboxes[:, 2] + reg_targets[:, 2] * w
    y2 = bboxes[:, 3] + reg_targets[:, 3] * h
    return np.stack([x1, y1, x2, y2], axis=1)

def square_boxes(boxes):
    """
    Convert rectangular boxes to square boxes by extending the shorter side.
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1, y1, x2, y2 = [boxes[:, i] for i in range(4)]
    w = x2 - x1
    h = y2 - y1
    max_side = np.maximum(w, h)
    cx = x1 + w / 2
    cy = y1 + h / 2
    nx1 = cx - max_side / 2
    ny1 = cy - max_side / 2
    nx2 = cx + max_side / 2
    ny2 = cy + max_side / 2
    return np.stack([nx1, ny1, nx2, ny2], axis=1)

def load_annotations(annotation_file_path):
    print(f"Simulating loading annotations from: {annotation_file_path}")
    sample_annotations = [
        {
            'image_path': 'path/to/image1.jpg',
            'boxes': np.array([[50, 50, 250, 250], [100, 100, 200, 200]]),
            'labels': np.array([1, 1]),
            'landmarks': np.array([[60, 60, 70, 70, 80, 80, 90, 90, 100, 100],
                                   [110, 110, 120, 120, 130, 130, 140, 140, 150, 150]])
        },
        {
            'image_path': 'path/to/image2.jpg',
            'boxes': np.array([[75, 75, 200, 200]]),
            'labels': np.array([1]),
            'landmarks': np.array([[80, 80, 90, 90, 100, 100, 110, 110, 120, 120]])
        }
    ]
    return sample_annotations
