import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxNet(nn.Module):
  def __init__(self, num_classes=10, input_size=(1, 32, 32, 32)):
    super(VoxNet, self).__init__()
    self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
    self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
    self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
    self.dropout1 = nn.Dropout(p=0.4)
    self.dropout2 = nn.Dropout(p=0.5)

    # Dynamically calculate flattened_size for fully connected layers
    # Pass a dummy tensor through the convolutional layers to get the shape
    with torch.no_grad():
        dummy_input = torch.zeros(input_size).unsqueeze(0) # Add batch dimension
        x = F.relu(self.conv1(dummy_input))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        flattened_size = torch.flatten(x, 1).shape[1]

    self.fc1 = nn.Linear(flattened_size, 128)
    self.fc2 = nn.Linear(128, num_classes)
    self.num_classes = num_classes

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = x.view(x.size(0), -1) # Flatten the tensor
    x = self.dropout1(x)
    x = F.relu(self.fc1(x))
    x = self.dropout2(x)
    x = self.fc2(x)
    return x