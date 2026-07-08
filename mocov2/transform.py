import torchvision.transforms as transforms

# Data Augmentations and Dataset/DataLoader Setup

class TwoCropsTransform:
    """A callable object that returns two augmented views of an image."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        q = self.transform(x) # Query view
        k = self.transform(x) # Key view
        return q, k

# Function to set up data transformations
def get_moco_transform(image_size=224):
    """Defines SimCLR-like data augmentations for MoCo v2."""
    augmentation_pipeline = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)), # Resize and crop randomly
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), # Random color jitter
        transforms.RandomGrayscale(p=0.2), # Random grayscale
        transforms.RandomHorizontalFlip(), # Random horizontal flip
        transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize ImageNet stats
    ])
    return TwoCropsTransform(augmentation_pipeline)