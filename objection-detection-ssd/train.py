import torch
from utils import generate_anchor_boxes,assign_targets_to_anchors

def train_model(num_epochs,model,optimizer,sdd_loss,num_classes_dataset,fm_sizes_for_anchors,scales,ars_per_fm,batch_size,device):
    print("Starting dummy training loop...")

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_epoch_loss = 0.0
        total_loc_loss = 0.0
        total_conf_loss = 0.0

        # Simulate iterating over a DataLoader
        # In a real scenario, this would be `for i, (images, gt_boxes_batch, gt_labels_batch) in enumerate(dataloader):`
        # We'll run a few dummy batches per epoch
        for i_batch in range(5): # Simulate 5 batches per epoch
            optimizer.zero_grad()

            # --- 1. Simulate Input Data and Ground Truth ---
            # For demonstration, generate random images and ground truth for a batch
            images = torch.randn(batch_size, 3, 300, 300).to(device)

            # Dummy Ground Truth for each image in the batch
            # Shape for each: (num_gt_objects_in_image, 4) and (num_gt_objects_in_image)
            batch_gt_boxes = []
            batch_gt_labels = []

            for _ in range(batch_size):
                num_gt_objects = torch.randint(1, 4, (1,)).item() # 1 to 3 objects per image
                # Random boxes: xmin, ymin, xmax, ymax (normalized [0,1])
                gt_boxes_img = torch.rand(num_gt_objects, 4)
                # Ensure xmax > xmin and ymax > ymin
                gt_boxes_img[:, 2:] = gt_boxes_img[:, :2] + torch.rand(num_gt_objects, 2) * 0.1 + 0.05 # small boxes
                gt_boxes_img = torch.clamp(gt_boxes_img, 0.0, 1.0)

                # Random labels: class 1 or 2 (0 is background)
                gt_labels_img = torch.randint(1, num_classes_dataset, (num_gt_objects,)).long()

                batch_gt_boxes.append(gt_boxes_img)
                batch_gt_labels.append(gt_labels_img)

            # --- 2. Model Forward Pass ---
            pred_cls, pred_reg = model(images)

            # --- 3. Target Encoding for the Batch ---
            # Need to iterate through each image in the batch to assign targets
            batch_encoded_targets = []
            batch_classification_targets = []

            # Regenerate anchors for each image if necessary, or once if fixed
            # Since anchors are fixed per image size, generate once if not dynamic
            current_anchors = generate_anchor_boxes(
                img_size=300, feature_map_sizes=fm_sizes_for_anchors,
                scales=scales, aspect_ratios=ars_per_fm, device=device
            )

            for img_idx in range(batch_size):
                gt_boxes_img = batch_gt_boxes[img_idx].to(device)
                gt_labels_img = batch_gt_labels[img_idx].to(device)

                encoded_t, classification_t, _ = assign_targets_to_anchors(
                    ground_truth_boxes=gt_boxes_img,
                    ground_truth_labels=gt_labels_img,
                    anchors=current_anchors,
                    num_classes=num_classes_dataset,
                    iou_threshold=0.5,
                    neg_pos_ratio=3,
                    device=device
                )
                batch_encoded_targets.append(encoded_t)
                batch_classification_targets.append(classification_t)

            # Stack encoded targets and classification targets for the batch
            batched_targets_cls = torch.stack(batch_classification_targets) # (batch_size, num_anchors)
            batched_targets_reg = torch.stack(batch_encoded_targets)      # (batch_size, num_anchors, 4)

            # --- 4. Calculate Loss ---
            total_loss, loc_loss, conf_loss = sdd_loss(
                pred_cls, pred_reg,
                batched_targets_cls, batched_targets_reg
            )

            # --- 5. Backpropagation and Optimization ---
            total_loss.backward()
            optimizer.step()

            total_epoch_loss += total_loss.item()
            total_loc_loss += loc_loss.item()
            total_conf_loss += conf_loss.item()

        # Print epoch statistics
        avg_total_loss = total_epoch_loss / (i_batch + 1)
        avg_loc_loss = total_loc_loss / (i_batch + 1)
        avg_conf_loss = total_conf_loss / (i_batch + 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {avg_total_loss:.4f}, Loc Loss: {avg_loc_loss:.4f}, Conf Loss: {avg_conf_loss:.4f}")
