def train_model(num_epochs,dataloader,optimizer,hqnn_model,cost_fn):
    losses = []
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: compute predicted output by HQNN
            predictions = hqnn_model(inputs)

            # Compute loss
            loss = cost_fn(predictions, targets)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Update model parameters
            optimizer.step()

            losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training loop finished.")
