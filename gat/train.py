def train(model,optimizer,data,criterion):
    model.train() # Set the model to training mode
    optimizer.zero_grad() # Clear gradients
    out = model(data.x, data.edge_index) # Perform a forward pass
    # Calculate loss using only training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward() # Perform a backward pass
    optimizer.step() # Update model parameters
    return loss.item()
