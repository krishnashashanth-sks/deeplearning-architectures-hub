import numpy as np
from PIL import Image
import os
import json

def create_dummy_coco_dataset(img_dir, annot_path, num_images=10, img_size=416, num_classes=80):
    images_info = {}
    annotations_list = []
    categories_list = []

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for i in range(num_images):
        img_id = str(i)
        file_name = f'image_{i:04d}.jpg'
        images_info[img_id] = {"id": img_id, "file_name": file_name, "width": img_size, "height": img_size}
        dummy_image = Image.fromarray(np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8))
        dummy_image.save(os.path.join(img_dir, file_name))

        for j in range(np.random.randint(1, 4)):
            ann_id = len(annotations_list)
            cat_id = np.random.randint(0, num_classes - 1) # Ensure valid class id
            x_min = np.random.uniform(0, img_size * 0.8)
            y_min = np.random.uniform(0, img_size * 0.8)
            w = np.random.uniform(img_size * 0.1, img_size * 0.2)
            h = np.random.uniform(img_size * 0.1, img_size * 0.2)
            annotations_list.append({
                "id": ann_id,
                "image_id": int(img_id),
                "category_id": cat_id,
                "bbox": [x_min, y_min, w, h],
                "area": w * h,
                "iscrowd": 0
            })

    for i in range(num_classes):
        categories_list.append({"id": i, "name": f"class_{i}", "supercategory": "none"})

    coco_format = {
        "images": images_info,
        "annotations": {str(ann['id']): ann for ann in annotations_list},
        "categories": categories_list
    }

    with open(annot_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"Created dummy dataset with {num_images} images and {len(annotations_list)} annotations.")