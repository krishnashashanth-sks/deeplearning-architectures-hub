from visualize import visualize_embeddings,visualize_graph
from model import GCN
import torch
from train import *
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
data = dataset[0]

visualize_graph(data)

model = GCN(num_node_features=data.num_node_features, num_classes=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Adam optimizer

# Train the model
for epoch in range(1, 201):
    loss = train(data,model,optimizer)
    # For simplicity, we'll report test accuracy directly after each epoch.
    # In a real scenario, you'd typically use a validation set for hyperparameter tuning
    # and only evaluate on the test set once at the end.
    test_acc =test(data,model)
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')

# Evaluate on the dedicated test set
final_accuracy = test(data,model)
print(f'\nFinal Test Accuracy: {final_accuracy:.4f}')
visualize_embeddings(data, model)