import torch.nn as nn
import torchvision.models as models

#  Base Encoder Network (e.g., ResNet)
class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        # Using ResNet50 as the base encoder
        resnet = models.resnet50(weights=None) # Start from scratch for self-supervised learning
        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = resnet.fc.in_features

    def forward(self, x):
        x = self.encoder(x)
        # Flatten the output for the projection head
        return x.view(x.size(0), -1)

# Projection Head (MLP)
class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.net(x)
