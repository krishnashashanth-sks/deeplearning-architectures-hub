import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale

# Define the number of points to sample per object
NUM_POINTS = 1024

# Define a transformation pipeline for preprocessing
# SamplePoints samples a fixed number of points
# NormalizeScale scales the points to fit within a unit sphere centered at the origin
pre_transform = SamplePoints(NUM_POINTS)

# Load the ModelNet40 dataset
# root specifies the directory where the dataset will be stored
# name='40' indicates ModelNet40
# train=True loads the training split
# pre_transform applies the defined preprocessing steps
train_dataset = ModelNet(root='./data/ModelNet40', name='40', train=True, pre_transform=pre_transform)
test_dataset = ModelNet(root='./data/ModelNet40', name='40', train=False, pre_transform=pre_transform)