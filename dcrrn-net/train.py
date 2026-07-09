def train_model(num_epochs,dcrnn_model,train_loader,optimizer,diffusion_matrix_f_on_device,diffusion_matrix_b_on_device,criterion,device):
    train_losses = []

    print("Starting DCRNN model training...")

    for epoch in range(num_epochs):
        dcrnn_model.train() # Set model to training mode
        epoch_batch_losses = []

        for batch_X, batch_y in train_loader:
            # Move data to the appropriate device (CPU/GPU)
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Zero out the gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = dcrnn_model(batch_X, diffusion_matrix_f_on_device, diffusion_matrix_b_on_device)

            # Calculate the loss
            loss = criterion(predictions, batch_y)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            epoch_batch_losses.append(loss.item())

        # Calculate average loss for the epoch
        avg_epoch_loss = sum(epoch_batch_losses) / len(epoch_batch_losses)
        train_losses.append(avg_epoch_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}')
    return train_losses