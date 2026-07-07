import torch

# --- Training and Evaluation Loop ---
def train(model, dataloader, optimizer, loss_fn, node_s_in_dim, node_v_in_dim, graph_v_out_dim, device):
    model.train()
    total_loss = 0
    for data in dataloader:
        # QM9 dataset stores atom features in data.x (scalar) and data.pos (coordinates for vectors)
        # We need to adapt data.x and data.pos to match our (s, v) input format
        # For demonstration, let's assume QM9.x as scalar features and QM9.pos as vector features for GVP.
        # In a real GVP-QM9 integration, you'd carefully map atomic numbers, charges, etc., to (s,v) features.

        # Dummy mapping for QM9 data to GVP (s,v) format for training example
        # data.x usually contains atom types (scalar)
        # data.pos contains 3D coordinates (can be used to derive vector features or directly as 3D vectors)
        # For now, let's simplify: use a slice of data.x for scalar and data.pos as 3D vectors.
        # NOTE: This is a placeholder. Proper GVP featurization for QM9 would be more involved.
        s_input = data.x[:, :node_s_in_dim] if data.x.shape[1] >= node_s_in_dim else torch.randn(data.num_nodes, node_s_in_dim)
        v_input = data.pos.view(data.num_nodes, -1, 3)[:, :node_v_in_dim, :] if data.pos.shape[1] >= node_v_in_dim else torch.randn(data.num_nodes, node_v_in_dim, 3)

        # Ensure the dimensions match the model's expected input dimensions
        if s_input.shape[1] != node_s_in_dim:
          s_input = torch.randn(data.num_nodes, node_s_in_dim)
        if v_input.shape[1] != node_v_in_dim:
          v_input = torch.randn(data.num_nodes, node_v_in_dim, 3)

        # Move data to device
        s_input, v_input = s_input.to(device), v_input.to(device)
        edge_index = data.edge_index.to(device)

        # QM9 targets are typically stored in data.y
        # Let's assume the first target in QM9.y corresponds to a scalar target
        # and we need to create a dummy vector target for the loss function.
        target_s = data.y[:, 0].to(device) # Example: using the first property as scalar target
        # For vector target, we'll need to define what it represents or create a dummy one for the loss.
        # For simplicity, create a dummy vector target for the loss to function.
        target_v = torch.randn(graph_v_out_dim, 3).unsqueeze(0).repeat(data.num_graphs, 1, 1).to(device) # Dummy for batch

        optimizer.zero_grad()
        out_s, out_v = model((s_input, v_input), edge_index)

        # Ensure output shapes match target shapes for loss calculation
        # out_s will be (graph_s_out_dim) for a single graph output from readout
        # out_v will be (graph_v_out_dim, 3)
        # We need to handle batching for the loss if DataLoader gives multiple graphs
        # For QM9, data.y[:,0] is (batch_size,)
        # We will assume graph_s_out_dim=1 and graph_v_out_dim=0 for simplicity here

        # Adjust targets for batch if `graph_out_dims` was used for readout
        if model.graph_out_dims is not None:
            # If readout is applied, outputs are (batch_size, graph_s_out_dim) and (batch_size, graph_v_out_dim, 3)
            # We need to reshape targets accordingly
            target_s_reshaped = target_s.unsqueeze(-1) # Correctly reshape (batch_size,) to (batch_size, 1)
            # Adjust target_v to match the batch size and dimensions of out_v
            target_v_reshaped = torch.randn_like(out_v) # Replace with actual target if available
        else:
            # If no readout, outputs are node-level and target should also be node-level
            target_s_reshaped = torch.randn_like(out_s)
            target_v_reshaped = torch.randn_like(out_v)

        loss = loss_fn(out_s, out_v, target_s_reshaped, target_v_reshaped)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
