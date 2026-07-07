import torch

def evaluate(model, dataloader, loss_fn, node_s_in_dim, node_v_in_dim, graph_v_out_dim, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            s_input = data.x[:, :node_s_in_dim] if data.x.shape[1] >= node_s_in_dim else torch.randn(data.num_nodes, node_s_in_dim)
            v_input = data.pos.view(data.num_nodes, -1, 3)[:, :node_v_in_dim, :] if data.pos.shape[1] >= node_v_in_dim else torch.randn(data.num_nodes, node_v_in_dim, 3)

            if s_input.shape[1] != node_s_in_dim:
              s_input = torch.randn(data.num_nodes, node_s_in_dim)
            if v_input.shape[1] != node_v_in_dim:
              v_input = torch.randn(data.num_nodes, node_v_in_dim, 3)

            s_input, v_input = s_input.to(device), v_input.to(device)
            edge_index = data.edge_index.to(device)

            target_s = data.y[:, 0].to(device)
            target_v = torch.randn(graph_v_out_dim, 3).unsqueeze(0).repeat(data.num_graphs, 1, 1).to(device) # Dummy for batch

            out_s, out_v = model((s_input, v_input), edge_index)

            if model.graph_out_dims is not None:
                target_s_reshaped = target_s.unsqueeze(-1) # Correctly reshape (batch_size,) to (batch_size, 1)
                target_v_reshaped = torch.randn_like(out_v)
            else:
                target_s_reshaped = torch.randn_like(out_s)
                target_v_reshaped = torch.randn_like(out_v)

            loss = loss_fn(out_s, out_v, target_s_reshaped, target_v_reshaped)
            total_loss += loss.item()
    return total_loss / len(dataloader)