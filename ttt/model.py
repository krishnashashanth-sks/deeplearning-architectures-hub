import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10, num_ss_classes=4): # num_ss_classes for 4 rotations (0, 90, 180, 270 degrees)
        super(CustomCNN, self).__init__()
        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Input channels 3 for RGB images
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classification Head (adjust input features based on your image size and feature extractor output)
        # Assuming input image size is something like 32x32, output from features will be 128 channels at 4x4
        # For a 32x32 image: 32 -> 16 -> 8 -> 4 (after 3 MaxPool2d with stride 2)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Self-Supervised Head (e.g., for rotation prediction)
        self.self_supervised_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_ss_classes)
        )

    def forward(self, x):
        # Pass through feature extractor
        features = self.features(x)

        # Flatten features for fully connected layers
        features = torch.flatten(features, 1) # Flatten all dimensions except batch

        # Classification task output
        cls_output = self.classifier(features);

        # Self-supervised task output
        ss_output = self.self_supervised_head(features)

        return cls_output, ss_output
