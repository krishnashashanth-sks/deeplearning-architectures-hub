from torch.utils.data import Dataset
import os
from PIL import Image
import json
import torch

class YOCONetDataset(Dataset):
    def __init__(self, img_dir, annot_path, img_size, anchors, num_classes, transform=None):
        self.img_dir = img_dir
        self.img_size = img_size
        self.anchors = anchors
        self.num_classes = num_classes
        self.transform = transform

        with open(annot_path, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = list(self.annotations['images'].keys())
        self.img_to_annots = {img_id: [] for img_id in self.image_ids}
        for ann_id, ann_data in self.annotations['annotations'].items():
            img_id = str(ann_data['image_id'])
            if img_id in self.img_to_annots:
                self.img_to_annots[img_id].append(ann_data)

        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.annotations['categories']}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.annotations['images'][img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size

        raw_annotations = self.img_to_annots.get(img_id, [])
        targets = []
        for ann in raw_annotations:
            bbox = ann['bbox'] # [x_min, y_min, width, height]
            category_id = ann['category_id']

            x_min, y_min, w, h = bbox
            x_center = (x_min + w / 2) / original_width
            y_center = (y_min + h / 2) / original_height
            w_norm = w / original_width
            h_norm = h / original_height

            targets.append([category_id, x_center, y_center, w_norm, h_norm])

        max_objects = 50
        if len(targets) > max_objects:
            targets = targets[:max_objects]
        else:
            while len(targets) < max_objects:
                targets.append([-1, 0., 0., 0., 0.])

        targets = torch.tensor(targets, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, targets