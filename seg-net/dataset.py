import fiftyone as fo
import fiftyone.zoo as foz

# Define the categories you want to use for training
# These will be mapped to class IDs 1 and 2, with 0 being background
selected_coco_categories = ['person', 'car']

# Map COCO category names to integer labels for the model
# 0 will be background, 1 for 'person', 2 for 'car'
category_to_label = {name: i + 1 for i, name in enumerate(selected_coco_categories)}
label_to_category = {i + 1: name for i, name in enumerate(selected_coco_categories)}

print(f"Mapping COCO categories to model labels: {category_to_label}")

# Load the COCO 2017 validation dataset
# We'll limit the max_samples for quicker demonstration
# The split='validation' is good for testing data loading quickly
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["segmentations"],
    classes=selected_coco_categories, # Only load segmentations for these classes
    max_samples=200 # Limiting to 200 samples for faster loading and testing
)

print(f"\nDataset loaded: {dataset}")
print(f"Number of samples: {len(dataset)}")