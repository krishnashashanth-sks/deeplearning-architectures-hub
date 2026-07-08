import torch
import torch.nn as nn

# RecursiveNN Class Definition
class RecursiveNN(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim=None):
    super(RecursiveNN, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.composition_layer = nn.Linear(hidden_dim * 2, hidden_dim)
    self.activation = nn.Tanh()
    # New: Layer to project leaf node features to hidden_dim
    self.leaf_projection_layer = nn.Linear(input_dim, hidden_dim)
    self.output_layer = None
    if output_dim is not None:
      self.output_layer = nn.Linear(hidden_dim, output_dim)

  def forward(self, node):
    if node.is_leaf():
      if node.feature_vector is None:
        raise ValueError("Leaf node must have a feature_vector initialized.")
      # Project leaf features before activation to match hidden_dim
      node.hidden_state = self.activation(self.leaf_projection_layer(node.feature_vector))
      return node.hidden_state

    child_hidden_states = []
    for child in node.children:
      child_hidden_states.append(self.forward(child))

    # Handle cases for 1 or 2 children
    if len(child_hidden_states) == 1:
      # If only one child, duplicate its hidden state to fit composition_layer's expected input
      combined_children = torch.cat([child_hidden_states[0], child_hidden_states[0]], dim=-1)
    elif len(child_hidden_states) == 2:
      combined_children = torch.cat(child_hidden_states, dim=-1)
    else:
      raise NotImplementedError("Current model only supports 1 or 2 children per node for simplicity.")

    node.hidden_state = self.activation(self.composition_layer(combined_children))
    return node.hidden_state

  def get_output(self, root_node_representation):
    if self.output_layer is not None:
      return self.output_layer(root_node_representation)
    else:
      return root_node_representation