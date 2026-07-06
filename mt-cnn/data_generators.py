import numpy as np
from utils import load_image,bbox_regression_target,iou,random_flip
import cv2

def pnet_data_generator(annotations, batch_size=64, img_size=(12, 12), p_neg=0.7):
    # Make a copy to shuffle, so the original list is not affected
    current_annotations = list(annotations)
    if not current_annotations:
        print("Warning: No annotations provided to P-Net generator. Yielding no data.")
        return # Exit generator if no data

    np.random.shuffle(current_annotations)
    annotation_ptr = 0

    images_batch = []
    cls_y_true_batch = []  # Will store [raw_label, one_hot_0, one_hot_1]
    bbox_y_true_batch = [] # Will store [raw_label, bbox_tx, bbox_ty, bbox_tw, bbox_th]

    while True: # Keep generating batches indefinitely
        # Accumulate samples until we have a full batch
        while len(images_batch) < batch_size:
            # If we've processed all annotations, reshuffle for next 'epoch'
            if annotation_ptr >= len(current_annotations):
                np.random.shuffle(current_annotations)
                annotation_ptr = 0
                # If after reshuffling, we still have no samples in batch (highly unlikely given dummy data),
                # but a check would be for a truly empty dataset
                if not current_annotations: # Should not happen if checked at start
                    print("Error: Annotations became empty during generation cycle.")
                    return

            img_data = current_annotations[annotation_ptr]
            image_path = img_data['image_path']
            gt_boxes = img_data['boxes'] # Assuming gt_boxes is a list/array of [x1, y1, x2, y2]

            original_image = load_image(image_path) # Uses the conceptual load_image function
            original_height, original_width, _ = original_image.shape

            current_image_samples_from_this_image = [] # Samples generated from the *current* image

            # 1. Generate Positive samples (IoU >= 0.65)
            for gt_box in gt_boxes:
                x1_gt, y1_gt, x2_gt, y2_gt = gt_box.astype(int)
                w_gt = x2_gt - x1_gt
                h_gt = y2_gt - y1_gt

                if w_gt > 0 and h_gt > 0: # Ensure valid bounding box dimensions
                    cropped_img = original_image[y1_gt:y2_gt, x1_gt:x2_gt].copy()
                    if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0: # Ensure crop is not empty
                        resized_img = cv2.resize(cropped_img, img_size)
                        target_bbox_values = bbox_regression_target(np.array([gt_box]), np.array([gt_box]))[0]
                        current_image_samples_from_this_image.append((resized_img, 1, target_bbox_values, [0, 1])) # Label 1: face, one-hot [0,1]

            # 2. Generate Partial samples (0.4 <= IoU < 0.65)
            for gt_box in gt_boxes:
                x1_gt, y1_gt, x2_gt, y2_gt = gt_box.astype(int)
                w_gt = x2_gt - x1_gt
                h_gt = y2_gt - y1_gt

                if w_gt > 0 and h_gt > 0: # Ensure valid bounding box dimensions
                    # Generate a slightly perturbed box to simulate a partial overlap
                    perturb_scale = 0.15 # Randomly shift/scale a bit
                    p_x1 = int(x1_gt + w_gt * np.random.uniform(-perturb_scale, perturb_scale))
                    p_y1 = int(y1_gt + h_gt * np.random.uniform(-perturb_scale, perturb_scale))
                    p_x2 = int(x2_gt + w_gt * np.random.uniform(-perturb_scale, perturb_scale))
                    p_y2 = int(y2_gt + h_gt * np.random.uniform(-perturb_scale, perturb_scale))

                    # Clip coordinates to image boundaries
                    p_x1, p_y1 = max(0, p_x1), max(0, p_y1)
                    p_x2, p_y2 = min(original_width, p_x2), min(original_height, p_y2)

                    if p_x2 > p_x1 + 5 and p_y2 > p_y1 + 5: # Ensure min size for cropped patch
                        prop_box = np.array([p_x1, p_y1, p_x2, p_y2])
                        iou_val = iou(prop_box, np.array([gt_box]))[0]

                        if 0.4 <= iou_val < 0.65: # IoU range for partial samples
                            cropped_img = original_image[p_y1:p_y2, p_x1:p_x2].copy()
                            if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0: # Ensure crop is not empty
                                resized_img = cv2.resize(cropped_img, img_size)
                                target_bbox_values = bbox_regression_target(np.array([prop_box]), np.array([gt_box]))[0]
                                current_image_samples_from_this_image.append((resized_img, -1, target_bbox_values, [0, 0])) # Label -1: partial, one-hot dummy [0,0]

            # 3. Generate Negative samples (IoU < 0.3)
            num_neg_per_image = 5 # Generate a fixed number of negatives per image for simulation
            for _ in range(num_neg_per_image):
                # Determine robust side for negative sample
                min_patch_side = max(1, min(original_width, original_height) // 5)
                max_patch_side = min(original_width, original_height) // 2

                if min_patch_side > max_patch_side:
                    continue # Cannot generate valid square patch with these min/max ratios

                side = np.random.randint(min_patch_side, max_patch_side + 1) # high is inclusive

                # Ensure valid range for x1, y1 coordinates
                max_nx1 = original_width - side
                max_ny1 = original_height - side

                if max_nx1 < 0 or max_ny1 < 0: # Check if patch can fit at all
                    continue # Skip if patch cannot fit

                neg_x1 = np.random.randint(0, max_nx1 + 1)
                neg_y1 = np.random.randint(0, max_ny1 + 1)

                neg_x2, neg_y2 = neg_x1 + side, neg_y1 + side
                neg_box = np.array([neg_x1, neg_y1, neg_x2, neg_y2])

                max_iou_gt = 0
                if len(gt_boxes) > 0: # Only calculate IoU if there are ground truth boxes
                    max_iou_gt = np.max(iou(neg_box, gt_boxes))

                if max_iou_gt < 0.3: # Ensure low IoU
                    cropped_img = original_image[neg_y1:neg_y2, neg_x1:neg_x2].copy()
                    if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0: # Ensure crop is not empty
                        resized_img = cv2.resize(cropped_img, img_size)
                        current_image_samples_from_this_image.append((resized_img, 0, np.zeros(4), [1, 0])) # Label 0: non-face, one-hot [1,0]

            # Add samples from this image to the overall batch accumulation lists
            for img_patch, raw_lbl, bbox_t, cls_oh in current_image_samples_from_this_image:
                # Apply random flip augmentation
                augmented_img, augmented_bbox_t, _ = random_flip(img_patch.copy(), bbox_t.copy(), None)
                # Normalize image pixels to [-1, 1]
                normalized_img = (augmented_img.astype(np.float32) - 127.5) / 127.5

                images_batch.append(normalized_img)
                # Construct y_true for classification loss: [raw_label, one_hot_class_0, one_hot_class_1]
                cls_y_true_batch.append(np.hstack([[raw_lbl], cls_oh]))
                # Construct y_true for bounding box regression loss: [raw_label, bbox_tx, bbox_ty, bbox_tw, bbox_th]
                bbox_y_true_batch.append(np.hstack([[raw_lbl], augmented_bbox_t]))

            annotation_ptr += 1

        # After the inner while loop finishes, images_batch should have at least `batch_size` samples (if generator is healthy).
        # Yield one full batch.
        # Only yield if there are samples in images_batch, to prevent 'list of length 0' error
        if len(images_batch) > 0: # This check is essential for robustness
            yield (np.array(images_batch[:batch_size], dtype=np.float32),
                   {
                       'flatten_cls': np.array(cls_y_true_batch[:batch_size], dtype=np.float32),
                       'flatten_reg': np.array(bbox_y_true_batch[:batch_size], dtype=np.float32)
                   })

            # Remove the yielded samples from the accumulation lists
            images_batch = images_batch[batch_size:]
            cls_y_true_batch = cls_y_true_batch[batch_size:]
            bbox_y_true_batch = bbox_y_true_batch[batch_size:]
        else:
            # This case should ideally not be reached if annotations are sufficient and generation logic is sound.
            # If reached, it means the generator could not produce any samples for a full batch.
            print("Warning: P-Net generator could not produce enough samples for a batch. Exiting generator.")
            return # Stop the generator if no data can be produced
