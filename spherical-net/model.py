import torch.nn as nn
from layers import SphericalConvLayer,SphericalPoolingLayer

class SCNNModel(nn.Module):
    def __init__(self, nside_initial, in_channels, num_classes):
        super(SCNNModel, self).__init__()
        self.nside_initial = nside_initial
        self.in_channels = in_channels
        self.num_classes = num_classes

        # --- Layer Block 1 ---
        # nside_initial -> nside_initial/2
        self.conv1 = SphericalConvLayer(in_channels, 32, nside=nside_initial)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = SphericalPoolingLayer(nside_initial, nside_initial // 2)

        # --- Layer Block 2 ---
        # nside_initial/2 -> nside_initial/4
        self.conv2 = SphericalConvLayer(32, 64, nside=nside_initial // 2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = SphericalPoolingLayer(nside_initial // 2, nside_initial // 4)

        # --- Fully Connected Layers ---
        # Calculate the number of pixels after pooling. Ensure it's valid for Healpix.
        self.nside_final = nside_initial // 4
        if self.nside_final < 1:
            raise ValueError("nside_initial is too small to perform two pooling steps.")
        self.num_pixels_final = 12 * self.nside_final**2

        self.fc1 = nn.Linear(64 * self.num_pixels_final, 128)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, in_channels, num_pixels)

        # --- Block 1 ---
        x = self.conv1(x)

        # BatchNorm1d expects (batch_size, channels, num_features). Our num_pixels acts as num_features.
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # --- Block 2 ---
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # --- Flatten and Fully Connected ---
        # Flatten the output for the fully connected layer
        # Current shape: (batch_size, out_channels_block2, num_pixels_final)
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch_size

        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
