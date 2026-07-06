def train_model(num_epochs,model,train_loader,optimizer,criterion,device):
    print(f"Starting supervised training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data # For supervised training, we only need inputs and labels
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            cls_outputs, _ = model(inputs) # Get classification outputs
            loss = criterion(cls_outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0: # Print every 100 mini-batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

    print("Finished supervised training.")