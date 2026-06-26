import torchvision
import torchvision.transforms as transforms

# 1. Define image transformations for the training set (with augmentation)
# Using common ImageNet mean and std for normalization as a placeholder
# ResNet models typically expect 224x224 input, so we'll resize here.
# CIFAR-10 images are 32x32, so we need significant resizing.
# A common practice is to resize to 256 then center crop to 224, or directly resize.
# Here, we'll resize to 224x224 directly for simplicity, then add augmentation.

transform_train = transforms.Compose([
    transforms.Resize(256), # Resize to 256 first
    transforms.RandomCrop(224, padding=4), # Then random crop to 224, with padding
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Define image transformations for the validation/test set (without augmentation)
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load the CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

# 4. Load the CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
