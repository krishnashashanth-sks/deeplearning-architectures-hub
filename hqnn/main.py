import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import HQNNLayer
from train import train_model

# Define the quantum device for simulation
dev = qml.device("default.qubit", wires=3)

# 1. Define a simple target dataset for demonstration
# Let's say we want the HQNN to output a specific expectation value vector
# For 3 qubits, the output of hqnn_layer_qnode is a list of 3 expectation values.
# Let's target [0.8, -0.6, 0.2] as an example.

# Create a dummy input for the HQNN layer, explicitly set to float64
# In a real scenario, this would come from preprocessing classical data
dummy_input = torch.tensor([0.5, 1.0, 1.5], requires_grad=True, dtype=torch.float64) # Example input features
target_output = torch.tensor([0.8, -0.6, 0.2], dtype=torch.float64)

# Wrap the input and target in a Dataset and DataLoader
dataset = TensorDataset(dummy_input.unsqueeze(0), target_output.unsqueeze(0))
dataloader = DataLoader(dataset, batch_size=1)

print("Defined dummy input and target output for training.")

# 2. Instantiate the HQNN model
n_qubits = 3
# Each qubit gets 3 rotation params (Rx, Ry, Rz) + 1 entanglement param (CRZ for cyclic)
# So, 3*3 (rotations) + 3 (CRZ) = 12 weights per layer
num_weights_per_layer = (3 * n_qubits) + n_qubits
hqnn_model = HQNNLayer(n_qubits=n_qubits, num_weights_per_layer=num_weights_per_layer)
print(f"Instantiated HQNN model with {n_qubits} qubits and {num_weights_per_layer} trainable weights.")

# 3. Define the Cost Function (Mean Squared Error for regression-like task)
# PyTorch's MSELoss works directly with tensors
cost_fn = nn.MSELoss()
print("Defined MSE Loss function.")

# 4. Define the Classical Optimizer
# We'll use Adam, a popular choice for deep learning models
optimizer = optim.Adam(hqnn_model.parameters(), lr=0.1) # Learning rate can be tuned
print("Defined Adam optimizer.")

# 5. Implement the Hybrid Quantum-Classical Iterative Training Loop
num_epochs = 50 # Number of training iterations

print("\nStarting hybrid quantum-classical training loop...")
train_model(num_epochs,dataloader,optimizer,hqnn_model,cost_fn)
# --- Inference/Evaluation ---
print("\n### Evaluating Trained HQNN Model ###")

# Get the final prediction from the trained model
# dummy_input needs to be unsqueezed to simulate a batch dimension of 1 for the model's forward method
final_prediction_eval = hqnn_model(dummy_input.unsqueeze(0))

print("\n--- Model Performance ---")
print("Target Output:", target_output.numpy())
print("Final Prediction:", final_prediction_eval.detach().numpy()[0]) # [0] to remove the batch dim for display
print(f"Final Loss (MSE): {cost_fn(final_prediction_eval, target_output.unsqueeze(0)).item():.6f}")

print("\nComplete HQNN implementation, training, and inference demonstrated successfully.")