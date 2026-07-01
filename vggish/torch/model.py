import torch.nn as nn
import torch.nn.functional as F

class VGGish(nn.Module):
    """
    PyTorch implementation of the VGGish architecture.
    The model expects a 4D tensor input of shape (batch_size, 1, 96, 64)
    representing log-mel-spectrograms.
    """
    def __init__(self):
        super(VGGish, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Fully Connected Layers
        # The output size after pooling for a 96x64 input is:
        # After pool1: (96/2)x(64/2) = 48x32
        # After pool2: (48/2)x(32/2) = 24x16
        # After pool3: (24/2)x(16/2) = 12x8
        # After pool4: (12/2)x(8/2) = 6x4
        # So, the input to the first FC layer is 512 * 6 * 4
        self.fc1 = nn.Linear(512 * 6 * 4, 4096)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, 128)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.pool4(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)

        return x