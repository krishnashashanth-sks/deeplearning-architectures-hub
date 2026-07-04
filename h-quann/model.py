import pennylane as qml
import torch.nn as nn

n_qubits = 2
n_layers = 4 # Number of strongly entangling layers

# Define a quantum device
dev = qml.device("default.qubit", wires=n_qubits, shots=None) # No shots for backprop

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Feature encoding: Encode classical data into quantum states
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # Variational quantum circuit (Ansatz): Parametrized quantum layers
    # Using StronglyEntanglingLayers for a more advanced ansatz
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # Measurement: Expectation value of PauliZ on the first qubit
    return qml.expval(qml.PauliZ(0))

# Define the weight shape for the StronglyEntanglingLayers
# The template expects a shape (n_layers, n_qubits, 3) for rotations
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

class HybridQuantumNeuralNetwork(nn.Module):
    def __init__(self,X_train):
        super().__init__()
        # Classical layer before the quantum circuit
        # Input features are 2, output to 2 qubits for encoding
        self.fc1 = nn.Linear(X_train.shape[1], n_qubits)

        # Quantum layer using PennyLane's TorchLayer
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # Classical layer after the quantum circuit
        # Quantum circuit outputs a single expectation value, map to 1 output for binary classification
        self.fc2 = nn.Linear(1, 1)

        # Activation function for output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Ensure input is a float tensor
        x = x.float()
        # Pass through the first classical layer
        x = self.fc1(x)
        # Pass through the quantum layer
        x = self.quantum_layer(x)
        # Reshape the output from quantum layer to (batch_size, 1) if it's a 1D tensor
        if x.dim() == 1:
            x = x.unsqueeze(1)
        # Pass through the second classical layer
        x = self.fc2(x)
        # Apply sigmoid activation
        x = self.sigmoid(x)
        return x

