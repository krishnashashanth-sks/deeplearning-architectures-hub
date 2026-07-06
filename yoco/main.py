import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader
from PIL import Image
import os
import shutil
from dataset import YOCONetDataset
from inference import decode_predictions,apply_nms,apply_confidence_threshold
from losses import Yocoloss
from model import YOCONetwork
from train import train_model
from utils import create_dummy_coco_dataset

# --- Configuration Parameters ---
NUM_ANCHORS = 3
NUM_CLASSES = 80 # Example for COCO dataset
IMAGE_SIZE = 416
# Example anchors, normalized to 0-1 scale relative to image size
ANCHORS = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32) / IMAGE_SIZE
LEARNING_RATE = 1e-4
EPOCHS = 5 # Reduced for demonstration purposes
LOG_INTERVAL = 1
CONF_THRESHOLD = 0.5 # Confidence threshold for filtering detections
NMS_IOU_THRESHOLD = 0.4 # IoU threshold for Non-Maximum Suppression

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Dummy Dataset Creation (for demonstration) ---
dummy_img_dir = './dummy_images'
dummy_annot_path = './dummy_annotations.json'

if not os.path.exists(dummy_annot_path) or not os.listdir(dummy_img_dir):
    if os.path.exists(dummy_img_dir):
        shutil.rmtree(dummy_img_dir)
    os.makedirs(dummy_img_dir)
    create_dummy_coco_dataset(dummy_img_dir, dummy_annot_path, num_images=10, img_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

# --- Transformations ---
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- DataLoaders ---
BATCH_SIZE = 4

train_dataset = YOCONetDataset(
    img_dir=dummy_img_dir,
    annot_path=dummy_annot_path,
    img_size=IMAGE_SIZE,
    anchors=ANCHORS,
    num_classes=NUM_CLASSES,
    transform=train_transform
)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True
)

val_dataset = YOCONetDataset(
    img_dir=dummy_img_dir,
    annot_path=dummy_annot_path,
    img_size=IMAGE_SIZE,
    anchors=ANCHORS,
    num_classes=NUM_CLASSES,
    transform=val_transform
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True
)

yoco_model = YOCONetwork(num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES).to(device)
yoco_loss_fn = Yocoloss(
    anchors=ANCHORS, num_classes=NUM_CLASSES, img_size=IMAGE_SIZE,
    ignore_threshold=0.5, lambda_coord=5.0, lambda_noobj=0.5
).to(device) # Loss functions are usually on CPU, but their internal tensors might need to be on device
optimizer = optim.Adam(yoco_model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
train_model(EPOCHS,yoco_model,train_dataloader,val_dataloader,optimizer,yoco_loss_fn,scheduler,LOG_INTERVAL,device)

# --- Inference Example ---
print("\n--- Running Inference Example ---")
yoco_model.eval() # Set model to evaluation mode

# Load and preprocess a sample image from the validation dataset
original_sample_idx = 0
original_image_path = os.path.join(
    val_dataset.img_dir,
    val_dataset.annotations['images'][val_dataset.image_ids[original_sample_idx]]['file_name']
)
original_image = Image.open(original_image_path).convert('RGB')

preprocessed_input_image = val_transform(original_image)
input_for_model = preprocessed_input_image.unsqueeze(0).to(device)

with torch.no_grad():
    raw_predictions = yoco_model(input_for_model)

print(f"Raw predictions shape: {raw_predictions.shape}")

# Post-process raw predictions
decoded_detections = decode_predictions(raw_predictions, ANCHORS, IMAGE_SIZE, NUM_CLASSES)
print(f"Shape after decoding: {decoded_detections.shape}")

filtered_detections_batch = apply_confidence_threshold(decoded_detections, CONF_THRESHOLD)
final_detections_batch = apply_nms(filtered_detections_batch, NMS_IOU_THRESHOLD, conf_threshold=CONF_THRESHOLD)

print(f"Number of final detections for the sample image: {len(final_detections_batch[0])}")
final_sample_detections = final_detections_batch[0].cpu().numpy()

print("Full YOCO implementation complete and example inference run.")
