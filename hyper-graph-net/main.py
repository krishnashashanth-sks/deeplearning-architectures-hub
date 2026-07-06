import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from model import HGNN
import torch
import torch.nn as nn

num_nodes = 10
num_hyperedges = 4
input_dim = 16
hidden_dim = 32
output_dim = 2

model_improved = HGNN(input_dim, hidden_dim, output_dim, num_nodes, num_hyperedges)

# Random node features
x = torch.randn((num_nodes, input_dim))

# Define hyperedge connections
# Example: node_to_hyperedge_index[0] = node IDs, [1] = hyperedge IDs
node_to_hyperedge_index = torch.tensor([
    [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9], # Node indices
    [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3]  # Hyperedge indices
], dtype=torch.long)

# The reverse mapping for the second pass (Hyperedge -> Node)
# Usually just the swap of the above
hyperedge_to_node_index = torch.stack([node_to_hyperedge_index[1], node_to_hyperedge_index[0]], dim=0)

# --- Training Loop ---

# Create dummy labels for node classification (e.g., 2 classes)
# Make sure labels match the num_nodes defined previously
y = torch.randint(0, output_dim, (num_nodes,), dtype=torch.long)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model_improved.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def train():
    model_improved.train()
    optimizer.zero_grad()
    out = model_improved(x, node_to_hyperedge_index, hyperedge_to_node_index)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

print("Starting training...")
for epoch in range(50): # Train for 50 epochs
    loss = train()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

print("\nTraining complete.")

# Evaluate the model (optional, but good practice)
model_improved.eval()
with torch.no_grad():
    final_out = model_improved(x, node_to_hyperedge_index, hyperedge_to_node_index)
    # For classification, you might want to get predictions
    predicted_classes = final_out.argmax(dim=1)
    accuracy = (predicted_classes == y).sum().item() / num_nodes
    print(f'Final Node Predictions: {predicted_classes}')
    print(f'True Node Labels:     {y}')
    print(f'Accuracy on synthetic data: {accuracy:.4f}')
# Combine predictions and true labels into a DataFrame for easy plotting
results_df = pd.DataFrame({
    'Node ID': range(num_nodes),
    'True Label': y.cpu().numpy(),
    'Predicted Label': predicted_classes.cpu().numpy()
})

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=results_df,
    x='Node ID',
    y='Predicted Label',
    hue='True Label',
    style='True Label',
    s=200, # size of points
    palette='deep',
    marker='o'
)

# Annotate points with their predicted label
for i, row in results_df.iterrows():
    plt.text(row['Node ID'] + 0.1, row['Predicted Label'] + 0.1, str(row['Predicted Label']),
             horizontalalignment='left', size='small', color='black', weight='semibold')

plt.title('Node Classification: Predicted vs. True Labels')
plt.xlabel('Node ID')
plt.ylabel('Label')
plt.yticks([0, 1]) # Assuming binary classification (0 or 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='True Label')
plt.tight_layout()
plt.show()