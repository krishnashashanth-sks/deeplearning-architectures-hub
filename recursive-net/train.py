import torch

def train_model(num_epochs,dataloader,optimizer,model,criterion):
    print("--- Starting Training Loop ---")
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (trees, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            # In this setup, `trees` will be a list containing one tree per batch
            tree = trees[0] # Get the single tree from the batch
            label = labels.squeeze(0) # Get the single label from the batch (remove batch dim)

            # Forward pass
            root_representation = model(tree)
            logits = model.get_output(root_representation)

            # Calculate loss
            loss = criterion(logits, label)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == label).sum().item()
            total_samples += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    print("--- Training Loop Completed ---")
