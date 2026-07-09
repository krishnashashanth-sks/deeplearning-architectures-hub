import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class DiffusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, K=2, bias=True, activation=F.relu):
        super(DiffusionConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.K = K
        self.activation = activation # Store activation function

        self.weight_f = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_b = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_f, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_b, a=np.sqrt(5))
        if self.bias is not None:
            # For linear layers, fan_in is typically the number of input features
            fan_in = self.in_channels
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X, diffusion_matrix_f, diffusion_matrix_b):
        # X: (num_nodes, in_channels)
        # diffusion_matrix_f/b: (num_nodes, num_nodes)

        diffusion_f = torch.matmul(diffusion_matrix_f, X)
        diffusion_b = torch.matmul(diffusion_matrix_b, X)

        # Apply weights for forward and backward diffusions
        output = torch.matmul(diffusion_f, self.weight_f) + torch.matmul(diffusion_b, self.weight_b)

        if self.bias is not None:
            output = output + self.bias

        # Apply activation if provided
        if self.activation is not None:
            output = self.activation(output)

        return output

class DCRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_nodes, K=2, bias=True):
        super(DCRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.K = K

        # Three DiffusionConv layers for update gate, reset gate, and candidate hidden state
        # in_channels for each is (input_size + hidden_size) because we concatenate X and H
        # out_channels for each is hidden_size
        # We set activation=None for these DiffusionConv layers as sigmoid/tanh will be applied externally.
        self.dconv_update_gate = DiffusionConv(
            in_channels=input_size + hidden_size,
            out_channels=hidden_size,
            num_nodes=num_nodes,
            K=K,
            bias=bias,
            activation=None # No activation inside DiffusionConv for gate outputs
        )
        self.dconv_reset_gate = DiffusionConv(
            in_channels=input_size + hidden_size,
            out_channels=hidden_size,
            num_nodes=num_nodes,
            K=K,
            bias=bias,
            activation=None # No activation inside DiffusionConv for gate outputs
        )
        # For the candidate state, its input depends on X and (r*H), and then tanh is applied.
        # So, the DiffusionConv should also return a linear output.
        self.dconv_candidate_state = DiffusionConv(
            in_channels=input_size + hidden_size, # This will be the size of [X, r*H]
            out_channels=hidden_size,
            num_nodes=num_nodes,
            K=K,
            bias=bias,
            activation=None # No activation inside DiffusionConv for candidate state output
        )

    def forward(self, X, H, diffusion_matrix_f, diffusion_matrix_b):
        # X: (num_nodes, input_size)
        # H: (num_nodes, hidden_size)

        # Concatenate current input X and previous hidden state H for gates
        # (num_nodes, input_size + hidden_size)
        combined_input_gates = torch.cat([X, H], dim=-1);

        # Calculate update gate
        # Apply DiffusionConv and then sigmoid activation
        z = torch.sigmoid(self.dconv_update_gate(combined_input_gates, diffusion_matrix_f, diffusion_matrix_b))

        # Calculate reset gate
        # Apply DiffusionConv and then sigmoid activation
        r = torch.sigmoid(self.dconv_reset_gate(combined_input_gates, diffusion_matrix_f, diffusion_matrix_b))

        # Calculate candidate hidden state
        # The input to the candidate state is X concatenated with (r * H)
        candidate_input_h_tilde = torch.cat([X, r * H], dim=-1)
        # Apply DiffusionConv and then tanh activation
        h_tilde = torch.tanh(self.dconv_candidate_state(candidate_input_h_tilde, diffusion_matrix_f, diffusion_matrix_b))

        # Calculate new hidden state using the GRU formula
        H_new = z * H + (1 - z) * h_tilde

        return H_new