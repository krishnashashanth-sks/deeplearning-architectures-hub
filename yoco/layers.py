import torch.nn as nn

# --- 1. CustomBackbone (Feature Extractor) ---

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='leaky_relu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class CustomBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super(CustomBackbone, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, stride=2, padding=1, activation='leaky_relu'),
            ConvBlock(32, 64, kernel_size=3, padding=1, stride=2, activation='leaky_relu'),
            ConvBlock(64, 128, kernel_size=3, padding=1, stride=1, activation='leaky_relu'),
            ConvBlock(128, 128, kernel_size=3, padding=1, stride=2, activation='leaky_relu'),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1, activation="leaky_relu"),
            ConvBlock(256, 256, kernel_size=3, stride=2, padding=1, activation="leaky_relu"),
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1, activation='leaky_relu'),
            ConvBlock(512, 512, kernel_size=3, stride=2, padding=1, activation='leaky_relu'),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1, activation='leaky_relu')
        )

    def forward(self, x):
        return self.features(x)


# --- 2. YOCONetworkHead (Detection Head) ---

class YOCONetworkHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(YOCONetworkHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.concentration_layers = nn.Sequential(
            ConvBlock(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, activation='leaky_relu'),
            ConvBlock(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, activation='leaky_relu'),
            ConvBlock(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, activation='leaky_relu'),
            ConvBlock(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, activation='leaky_relu')
        )
        self.prediction_layer = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.concentration_layers(x)
        out = self.prediction_layer(x)
        batch_size, channels, grid_h, grid_w = out.shape
        out = out.view(batch_size, self.num_anchors, (5 + self.num_classes), grid_h, grid_w)
        return out.permute(0, 1, 3, 4, 2).contiguous()
