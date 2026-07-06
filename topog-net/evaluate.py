import torch

def evaluate_model(model,loader,loss_fn,device):
    model.eval() # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad(): # Disable gradient computation
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.x_topo, data.batch)
            loss = loss_fn(out, data.y.squeeze(1))
            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)