from visualize import visualize_embeddings_gat,visualize_graph
from model import GAT
import torch
from train import *
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
data = dataset[0]

visualize_graph(data)

gat_model = GAT(num_node_features=data.num_node_features, num_classes=dataset.num_classes)

optimizer_gat = torch.optim.Adam(gat_model.parameters(), lr=0.01, weight_decay=5e-4)

# Train the GAT model
for epoch in range(1, 201):
    loss_gat = train_gat(data,gat_model,optimizer_gat)
    test_acc_gat = test_gat(gat_model,data)
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss_gat:.4f}, Test Acc: {test_acc_gat:.4f}')

# Evaluate on the dedicated test set
final_accuracy_gat = test_gat(gat_model,data)
print(f'\nGraph Attention Network Final Test Accuracy: {final_accuracy_gat:.4f}')
visualize_embeddings_gat(data, gat_model)