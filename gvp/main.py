from rdkit import Chem
import py3Dmol
import numpy as np
from model import GVPGNNStack
from losses import GVPLoss
import torch
from torch.utils.data import DataLoader
import torch_geometric.datasets as datasets
from train import train
from evaluate import evaluate

# Node feature dimensions
node_s_in_dim = 16
node_v_in_dim = 4

node_s_hidden_dim = 32
node_v_hidden_dim = 8

node_s_out_dim = 8 # These are the node features after GNN layers
node_v_out_dim = 2

# Graph-level output dimensions (for readout)
graph_s_out_dim = 1 # Changed from 4 to 1 to match single scalar target
graph_v_out_dim = 1

# Message (or Attention Head) feature dimensions
msg_s_dim = 8 # Used by GVPGNNLayer
msg_v_dim = 2 # Used by GVPGNNLayer
head_s_dim = 8 # Used by GVPAttentionLayer (Q,K,V dimensions)
head_v_dim = 2 # Used by GVPAttentionLayer (Q,K,V dimensions)

num_gnn_layers = 3 # Let's stack 3 GNN layers

gvp_model = GVPGNNStack(
    num_layers=num_gnn_layers,
    node_in_dims=(node_s_in_dim, node_v_in_dim),
    node_hidden_dims=(node_s_hidden_dim, node_v_hidden_dim),
    node_out_dims=(node_s_out_dim, node_v_out_dim), # Output of GNN layers (node-level)
    msg_dims=(msg_s_dim, msg_v_dim), # Still needed by some default args, but attention layers will use head_dims
    graph_out_dims=(graph_s_out_dim, graph_v_out_dim),
    layer_type='Attention', # Specify 'Attention' layers
    head_dims=(head_s_dim, head_v_dim) # Define dimensions for attention heads
)

loss_fn = GVPLoss() # Using the previously defined custom GVP loss
optimizer = torch.optim.Adam(gvp_model.parameters(), lr=0.001)

# --- Setup for QM9 Data ---
# The QM9 dataset contains 130,831 molecules with 19 properties
# For this example, we'll use a subset for faster demonstration

dataset = datasets.QM9(root='/tmp/QM9') # Already downloaded from previous step

# Split dataset into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

batch_size_train = 32 # Can be adjusted
batch_size_eval = 64  # Can be adjusted

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gvp_model.to(device)

num_epochs = 5 # Number of training epochs

print(f"\nStarting training on {device} for {num_epochs} epochs...")
for epoch in range(num_epochs):
    train_loss = train(gvp_model, train_loader, optimizer, loss_fn, node_s_in_dim, node_v_in_dim, graph_v_out_dim, device)
    val_loss = evaluate(gvp_model, val_loader, loss_fn, node_s_in_dim, node_v_in_dim, graph_v_out_dim, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print("\nTraining complete! Evaluating on test set...")
test_loss = evaluate(gvp_model, test_loader, loss_fn, device)
print(f'Test Loss: {test_loss:.4f}')

print("\nThis demonstrates a complete (though simplified) training and evaluation pipeline for a GVP-GNN model using a real-world dataset like QM9.")
print("Note: The featurization of QM9 data into (s,v) for GVP input and the definition of vector targets are simplified for demonstration. In a production setting, these would require careful design based on the specific task (e.g., predicting molecular properties that are scalars, vectors, or tensors).")
# --- Model Inference ---
gvp_model.eval() # Set model to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gvp_model.to(device)

print("\n--- Performing Model Inference and Visualization ---")

# Get one sample batch from the test set
data = next(iter(test_loader))
data = data.to(device) # Move entire data object to device

s_input = data.x[:, :node_s_in_dim] if data.x.shape[1] >= node_s_in_dim else torch.randn(data.num_nodes, node_s_in_dim, device=device)
v_input = data.pos.view(data.num_nodes, -1, 3)[:, :node_v_in_dim, :] if data.pos.shape[1] >= node_v_in_dim else torch.randn(data.num_nodes, node_v_in_dim, 3, device=device)

# Ensure the dimensions match the model's expected input dimensions
# (These checks might be redundant if the dataset featurization is robust)
if s_input.shape[1] != node_s_in_dim:
    s_input = torch.randn(data.num_nodes, node_s_in_dim, device=device)
if v_input.shape[1] != node_v_in_dim:
    v_input = torch.randn(data.num_nodes, node_v_in_dim, 3, device=device)

edge_index = data.edge_index.to(device)

with torch.no_grad():
    output_s, output_v = gvp_model((s_input, v_input), edge_index)

# --- Visualization (Basic) ---
print(f"\nInference on a single batch from the test set (batch_size={data.num_graphs}):")
print(f"Predicted Scalar Output Shape: {output_s.shape}")
print(f"Predicted Vector Output Shape: {output_v.shape}")

# Let's take the first sample in the batch for detailed display
sample_idx = 0

predicted_scalar_val = output_s[sample_idx].item()
# Squeeze to get a (3,) vector if graph_v_out_dim is 1, otherwise it will be (graph_v_out_dim, 3)
predicted_vector_val = output_v[sample_idx].squeeze(0).cpu().numpy()

# Ground truth for scalar property (first property from QM9 data.y)
# data.y is (batch_size, num_properties)
true_scalar_val = data.y[sample_idx, 0].item()

# --- Molecular Visualization ---

single_graph_data = data.get_example(sample_idx)

atomic_numbers = single_graph_data.x[:, 0].cpu().numpy().astype(int)
positions = single_graph_data.pos.cpu().numpy()

# Create an RDKit molecule object
# First, create an EditableMol (Editable Molecule) and add atoms
emol = Chem.EditableMol(Chem.Mol())
for atom_num in atomic_numbers:
    atom = Chem.Atom(int(atom_num))
    emol.AddAtom(atom)

# Create a molecule from the editable molecule
rdkit_mol = emol.GetMol()

# Add conformation (3D coordinates) to the molecule
conf = Chem.Conformer(rdkit_mol.GetNumAtoms())
for i in range(rdkit_mol.GetNumAtoms()):
    conf.SetAtomPosition(i, positions[i].tolist()) # positions is (num_atoms, 3)
rdkit_mol.AddConformer(conf)

# Generate a 3D view using py3Dmol
view = py3Dmol.view(width=400, height=300)

# Add the molecule to the viewer
# Py3Dmol can directly take an RDKit molecule object
view.addModel(Chem.MolToMolBlock(rdkit_mol), 'mol')
view.setStyle({'stick':{}, 'sphere':{'scale':0.3}})
view.zoomTo()
view.show()

print(f"\nDisplayed molecular structure for sample {sample_idx}.")

vector_origin = np.mean(positions, axis=0) # e.g., molecule's center of mass
vector_direction = predicted_vector_val # From previous inference step

view.addArrow({
    'start': {'x': float(vector_origin[0]), 'y': float(vector_origin[1]), 'z': float(vector_origin[2])},
    'end': {
        'x': float(vector_origin[0] + vector_direction[0]),
        'y': float(vector_origin[1] + vector_direction[1]),
        'z': float(vector_origin[2] + vector_direction[2])
    },
    'radius': 0.1,
    'color': 'blue'
})
view.update()
view.show()