import torch

def train(num_epochs,model,train_loader,device,optimizer,criterion,test_loader):
    print(f"\nStarting conceptual training loop for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (i+1) % 100 == 0: # Print every 100 mini-batches
                print(f'  Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / total_samples
        epoch_accuracy = 100 * correct_predictions / total_samples
        print(f'Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%')

        # (Optional) Validation step after each epoch
        model.eval() # Set model to evaluation mode
        val_correct_predictions = 0
        val_total_samples = 0
        val_running_loss = 0.0
        # Note: In a real scenario, use torch.no_grad() for validation
        # For this conceptual step, we omit it as per instructions but note its importance.
        # with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total_samples += labels.size(0)
            val_correct_predictions += (predicted == labels).sum().item()

        val_loss = val_running_loss / val_total_samples
        val_accuracy = 100 * val_correct_predictions / val_total_samples
        print(f'  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
