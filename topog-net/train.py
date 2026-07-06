def train_model(model,train_loader,optimizer,loss_fn,device):
    model.train() # Set the model to training mode
    total_loss = 0
    for data in train_loader:
        # Move data to the appropriate device
        data = data.to(device)

        optimizer.zero_grad() # Clear gradients

        # Forward pass
        # Ensure x_topo is handled correctly. If it's graph-level, it's (batch_size, num_topo_features).
        # The model's forward method expects it this way.
        out = model(data.x, data.edge_index, data.x_topo, data.batch)

        # QM9 has 19 regression targets, stored in data.y. y is (batch_size, 1, 19). Squeeze to (batch_size, 19).
        loss = loss_fn(out, data.y.squeeze(1)) # Calculate loss

        loss.backward() # Perform backpropagation
        optimizer.step() # Update model parameters

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)