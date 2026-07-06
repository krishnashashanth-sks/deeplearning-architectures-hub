import torch
from gudhi.representations import PersistenceImage
import numpy as np
import gudhi as gd

# Function to extract topological features for a single graph
def extract_topological_features(data_graph, dim_max=2, resolution=(20, 20)):
    # data_graph is a torch_geometric.data.Data object
    # We use node positions (pos) to build a Vietoris-Rips complex
    points = data_graph.pos.numpy()

    if len(points) < 2: # Rips complex requires at least 2 points
        # Return a zero vector if not enough points
        pi_dim = resolution[0] * resolution[1]
        return torch.zeros(2 * pi_dim)

    # Build Rips Complex from points
    # max_edge_length needs to be chosen carefully. This is a heuristic value.
    max_filtration_value = 5.0 # Max distance for creating edges in Rips complex

    rips_complex = gd.RipsComplex(points=points, max_edge_length=max_filtration_value)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dim_max)

    # Compute persistence
    simplex_tree.compute_persistence()
    persistence = simplex_tree.persistence()

    # Separate persistence diagrams for H0 and H1
    # The birth and death values are typically (birth, death). Death can be inf.
    diag_h0_raw = [p[1] for p in persistence if p[0] == 0]
    diag_h1_raw = [p[1] for p in persistence if p[0] == 1]

    # Initialize PersistenceImage without inf_value argument
    p_img = PersistenceImage(resolution=resolution)

    # Helper function to process raw diagrams and handle infinite values
    def process_diagram(raw_diagram, max_val):
        if not raw_diagram:
            return np.array([])
        diagram_np = np.array(raw_diagram)
        # Replace infinite death values with max_val
        diagram_np[np.isinf(diagram_np[:, 1]), 1] = max_val
        return diagram_np

    diag_h0_processed = process_diagram(diag_h0_raw, max_filtration_value)
    diag_h1_processed = process_diagram(diag_h1_raw, max_filtration_value)

    # Process H0 diagram
    if diag_h0_processed.size > 0:
        pi_h0 = p_img.fit_transform([diag_h0_processed])[0]
    else:
        pi_h0 = np.zeros(resolution[0] * resolution[1])

    # Process H1 diagram
    if diag_h1_processed.size > 0:
        pi_h1 = p_img.fit_transform([diag_h1_processed])[0]
    else:
        pi_h1 = np.zeros(resolution[0] * resolution[1])

    # Concatenate and convert to tensor
    topo_features = np.concatenate((pi_h0, pi_h1))
    return torch.tensor(topo_features, dtype=torch.float)
