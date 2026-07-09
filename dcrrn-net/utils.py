import torch
import numpy as np

def compute_scaled_diffusion_matrix(adj_matrix, k=2): # k is the diffusion step
    """
    Computes scaled forward and backward diffusion matrices.
    Args:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        k (int): Number of diffusion steps (powers of transition matrix).
    Returns:
        tuple: (scaled_diffusion_matrix_f, scaled_diffusion_matrix_b)
    """
    num_nodes = adj_matrix.shape[0]

    # Add self-loops to adjacency matrix for a robust diffusion process
    adj_matrix_with_self_loops = adj_matrix + np.eye(num_nodes)

    # Calculate out-degree and in-degree matrices
    out_degree_matrix = np.diag(1 / np.sum(adj_matrix_with_self_loops, axis=1))
    in_degree_matrix = np.diag(1 / np.sum(adj_matrix_with_self_loops, axis=0))

    # Calculate forward transition matrix (P_f = D_out^-1 * A)
    P_f = np.dot(out_degree_matrix, adj_matrix_with_self_loops)

    # Calculate backward transition matrix (P_b = D_in^-1 * A^T)
    P_b = np.dot(in_degree_matrix, adj_matrix_with_self_loops.T)

    # Initialize scaled diffusion matrices as identity matrices
    scaled_diffusion_matrix_f = np.eye(num_nodes)
    scaled_diffusion_matrix_b = np.eye(num_nodes)

    # Accumulate powers of transition matrices
    current_P_f = np.eye(num_nodes)
    current_P_b = np.eye(num_nodes)

    for _ in range(k):
        current_P_f = np.dot(current_P_f, P_f)
        current_P_b = np.dot(current_P_b, P_b)
        scaled_diffusion_matrix_f += current_P_f
        scaled_diffusion_matrix_b += current_P_b

    # Normalize the scaled matrices (divide by k+1, as it includes the 0-th power)
    scaled_diffusion_matrix_f /= (k + 1)
    scaled_diffusion_matrix_b /= (k + 1)

    # Convert to PyTorch tensors
    scaled_diffusion_matrix_f = torch.tensor(scaled_diffusion_matrix_f, dtype=torch.float32)
    scaled_diffusion_matrix_b = torch.tensor(scaled_diffusion_matrix_b, dtype=torch.float32)

    print(f"Computed scaled forward diffusion matrix shape: {scaled_diffusion_matrix_f.shape}")
    print(f"Computed scaled backward diffusion matrix shape: {scaled_diffusion_matrix_b.shape}")

    return scaled_diffusion_matrix_f, scaled_diffusion_matrix_b
