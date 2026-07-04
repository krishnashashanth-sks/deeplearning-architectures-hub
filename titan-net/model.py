import torch.nn as nn
from layers import InitialStem,Bottleneck
import torch

class TitanNet(nn.Module):
    expansion = 4

    def __init__(self, block, num_blocks, num_classes=10):
        super(TitanNet, self).__init__()
        self.in_channels = 64

        self.initial_stem = InitialStem(in_channels=3)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512 * block.expansion, 1000)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(1000)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(1000, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.initial_stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.gelu(out)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        out = self.softmax(out)
        return out

# Helper function to create a TitanNet model
def TitanNet50(num_classes=10):
    return TitanNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
