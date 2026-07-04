def train_lista_net(model, optimizer, loss_fn, data_loader, num_epochs, device):
    model.train() # Set the model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (y_batch, x_true_batch) in enumerate(data_loader):
            y_batch = y_batch.to(device)
            x_true_batch = x_true_batch.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            x_pred_batch = model(y_batch)

            # Calculate loss
            loss = loss_fn(x_pred_batch, x_true_batch)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    print("Training complete.")