import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import AdvancedSSD
from losses import SSDLoss
from train import train_model
from utils import generate_anchor_boxes
from inference import post_process_detections
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# --- 1. Real Custom Dataset Definition ---

from dataset import RealObjectDetectionDataset

# --- 2. Configurations & Structural Setup ---
num_classes_dataset = 3  # Set this to total actual classes + 1 (for background class 0)
batch_size = 4
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define image parsing transforms tailored for the SSD expected architecture size (300x300)
data_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths pointing to actual storage configurations
TRAIN_IMG_DIR = "path/to/your/train/images"
#Use the same pattern i have given in data folder
TRAIN_ANNO_FILE = "data/train_annotations.json"

TEST_IMG_DIR = "path/to/your/test/images"
TEST_ANNO_FILE = "path/to/your/test_annotations.json"

# Initialize DataLoaders via Custom Collate Function
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    gt_boxes_batch = [item[1] for item in batch]
    gt_labels_batch = [item[2] for item in batch]
    return images, gt_boxes_batch, gt_labels_batch

train_dataset = RealObjectDetectionDataset(TRAIN_IMG_DIR, TRAIN_ANNO_FILE, transform=data_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

test_dataset = RealObjectDetectionDataset(TEST_IMG_DIR, TEST_ANNO_FILE, transform=data_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


# --- 3. Anchor Configuration and Initialization ---
fm_sizes_for_anchors = [(10, 10), (5, 5), (3, 3), (2, 2), (2, 2)]
min_scale = 0.1
max_scale = 0.9
scales = [min_scale + (max_scale - min_scale) * k / (len(fm_sizes_for_anchors) - 1) for k in range(len(fm_sizes_for_anchors))]
common_aspect_ratios = [2.0, 3.0]
ars_per_fm = [common_aspect_ratios for _ in fm_sizes_for_anchors]


# --- 4. Model Instantiation & Training Execution ---
model = AdvancedSSD(backbone_name='resnet50', num_classes=num_classes_dataset).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
sdd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

print("\n--- Initiating Model Training Sequence ---")
train_model(num_epochs, model, optimizer, sdd_loss, num_classes_dataset, fm_sizes_for_anchors, scales, ars_per_fm, batch_size, device)


# --- 5. COCO Validation Evaluation ---
print("\n--- Running COCO Evaluation on Test Split ---")
collected_predictions = []
collected_ground_truths = []

model.eval()
with torch.no_grad():
    for i_batch, (images, gt_boxes_batch, gt_labels_batch) in enumerate(test_dataloader):
        images = images.to(device)
        pred_cls, pred_reg = model(images)

        for i in range(images.size(0)):
            current_image_anchors = generate_anchor_boxes(
                img_size=300, feature_map_sizes=fm_sizes_for_anchors,
                scales=scales, aspect_ratios=ars_per_fm, device=device
            )

            final_boxes, final_labels, final_scores = post_process_detections(
                pred_cls[i], pred_reg[i], current_image_anchors,
                num_classes=num_classes_dataset,
                conf_threshold=0.01,
                nms_iou_threshold=0.45,
                top_k=200
            )

            global_img_id = i_batch * test_dataloader.batch_size + i
            collected_predictions.append({
                'image_id': global_img_id,
                'boxes': final_boxes.cpu().numpy(),
                'labels': final_labels.cpu().numpy(),
                'scores': final_scores.cpu().numpy()
            })
            collected_ground_truths.append({
                'image_id': global_img_id,
                'boxes': gt_boxes_batch[i].cpu().numpy(),
                'labels': gt_labels_batch[i].cpu().numpy()
            })

# Transform data structures into formal COCO indices
coco_gt_dataset = {"images": [], "annotations": [], "categories": []}
for class_id in range(1, num_classes_dataset):
    coco_gt_dataset["categories"].append({"id": class_id, "name": f"class_{class_id}"})

annotation_id_counter = 0
for gt_data in collected_ground_truths:
    image_id = gt_data['image_id']
    coco_gt_dataset['images'].append({"id": image_id, "width": 300, "height": 300})
    for box, label in zip(gt_data['boxes'], gt_data['labels']):
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        coco_gt_dataset['annotations'].append({
            "id": annotation_id_counter,
            "image_id": image_id,
            "category_id": int(label),
            "bbox": [xmin * 300, ymin * 300, width * 300, height * 300],  # Denormalize to pixel scale for COCOeval
            "area": (width * 300) * (height * 300),
            "iscrowd": 0
        })
        annotation_id_counter += 1

coco_gt = COCO()
coco_gt.dataset = coco_gt_dataset
coco_gt.createIndex()

coco_dt_list = []
for pred_data in collected_predictions:
    image_id = pred_data['image_id']
    for box, label, score in zip(pred_data['boxes'], pred_data['labels'], pred_data['scores']):
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        coco_dt_list.append({
            "image_id": image_id,
            "category_id": int(label),
            "bbox": [xmin * 300, ymin * 300, width * 300, height * 300],
            "score": float(score)
        })

with open('real_detections.json', 'w') as f:
    json.dump(coco_dt_list, f)

if len(coco_dt_list) > 0:
    coco_dt = coco_gt.loadRes('real_detections.json')
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
else:
    print("No test detections returned to match against ground truth labels.")