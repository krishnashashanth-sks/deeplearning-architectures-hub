import torch

def train(num_epochs,model,train_dataloader,optimizer,criterion,device):
    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        model.train() # Set model to training mode
        current_train_loss = 0.0
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            current_train_loss += loss.item()

        avg_train_loss = current_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Loop ---
        model.eval() # Set model to evaluation mode
        current_val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation for validation
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                current_val_loss += loss.item()

        avg_val_loss = current_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}, "
            f"Time: {epoch_duration:.2f}s")

    print("Training complete!")
    return train_losses,val_losses