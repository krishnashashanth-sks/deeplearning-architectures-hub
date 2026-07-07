from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import json

class RealObjectDetectionDataset(Dataset):
    """
    A standard PyTorch dataset layout for real object detection imagery.
    Assumes a dataset directory structure containing an image folder and a JSON annotation map.
    """
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Expecting a JSON structure matching or mapping to image names and coordinate arrays
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.image_keys = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        img_name = self.image_keys[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load and convert image to RGB
        image = Image.open(img_path).convert("RGB")
        
        # Extract metadata arrays
        anno_data = self.annotations[img_name]
        # Coordinates must be in normalized format [xmin, ymin, xmax, ymax] scaled between [0.0, 1.0]
        boxes = torch.tensor(anno_data['boxes'], dtype=torch.float32)
        labels = torch.tensor(anno_data['labels'], dtype=torch.long) # Background is Class 0

        if self.transform:
            image = self.transform(image)
            
        return image, boxes, labels

