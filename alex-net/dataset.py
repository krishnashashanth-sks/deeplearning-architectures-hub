import torchvision
import torchvision.transforms as transforms

IMG_SIZE = 227 # AlexNet typically uses 227x227 or 224x224. Using 227 for this example.
MEAN = [0.4914, 0.4822, 0.4465] # CIFAR-10 mean values
STD = [0.2023, 0.1994, 0.2010]  # CIFAR-10 standard deviation values

# 2. Define a set of transformations for the training dataset.
# Including resizing, data augmentation (random crop, random horizontal flip), and normalization.
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# 3. Define a separate set of transformations for the validation/test dataset.
# Including resizing and normalization, without data augmentation.
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# 4. Load the chosen dataset (CIFAR-10), applying the defined transformations.
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)