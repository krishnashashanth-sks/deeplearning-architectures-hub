from main import qml,dev
import torch.nn as nn
import torch
from pennylane import numpy as np

@qml.qnode(dev)
def hqnn_layer_qnode(inputs, weights):
    """
    A single HQNN layer as a PennyLane QNode.
    It applies initial encoding based on inputs and then parameterized rotation and entanglement blocks.
    """
    n_qubits = 3
    param_idx = 0

    # 1. Input Data Encoding (Angle Encoding - Rx gates)
    # Map classical input features to quantum states
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)

    # 2. Parameterized Rotation Block (Rx, Ry, Rz for each qubit)
    # These are the trainable parameters of the layer
    for i in range(n_qubits):
        qml.RX(weights[param_idx], wires=i)
        param_idx += 1
        qml.RY(weights[param_idx], wires=i)
        param_idx += 1
        qml.RZ(weights[param_idx], wires=i)
        param_idx += 1

    # 3. Parameterized Entanglement Block (Cyclic CRZ)
    # Introduces entanglement between qubits with trainable parameters
    for i in range(n_qubits):
        control_qubit = i
        target_qubit = (i + 1) % n_qubits
        qml.CRZ(weights[param_idx], wires=[control_qubit, target_qubit])
        param_idx += 1

    # Output Measurement: Expectation value of PauliZ on each qubit
    # This allows us to extract classical information for the next layer or output
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class HQNNLayer(nn.Module):
  def __init__(self,n_qubits=3,num_weights_per_layer=12):
    super().__init__()
    self.n_qubits=n_qubits
    self.num_weights_per_layer=num_weights_per_layer
    # Initialize weight tensor with float64 using torch.empty
    weight_tensor=torch.empty(num_weights_per_layer, dtype=torch.float64)
    nn.init.uniform_(weight_tensor,-np.pi,np.pi)
    self.q_weights=nn.Parameter(weight_tensor)
    self.q_node=hqnn_layer_qnode
  def forward(self,inputs):
    # Ensure inputs are float64. unsqueeze(0) ensures a batch dimension of 1 if inputs is a single data point.
    inputs=inputs.double()
    # Call the QNode with the first element of the batch for processing
    q_output=self.q_node(inputs[0],self.q_weights)
    # Stack the QNode outputs and add an explicit batch dimension
    return torch.stack(q_output).unsqueeze(0)

