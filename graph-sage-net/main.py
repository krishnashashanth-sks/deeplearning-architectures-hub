from visualize import visualize_embeddings_sage,visualize_graph
from model import GraphSAGE
import torch
from train import *
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
data = dataset[0]

visualize_graph(data)

graphsage_model = GraphSAGE(num_node_features=data.num_node_features, num_classes=dataset.num_classes)

optimizer_sage=torch.optim.Adam(graphsage_model.parameters(),lr=0.01)

# Train the GraphSAGE model
for epoch in range(1, 201):
    loss_sage = train_sage(data,graphsage_model,optimizer_sage)
    test_acc_sage = test_sage(data,graphsage_model)
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss_sage:.4f}, Test Acc: {test_acc_sage:.4f}')

# Evaluate on the dedicated test set
final_accuracy_sage = test_sage(data,graphsage_model)
print(f'\nGraphSAGE Final Test Accuracy: {final_accuracy_sage:.4f}')
visualize_embeddings_sage(data, graphsage_model)