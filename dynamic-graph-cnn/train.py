import torch

def train(model, train_loader, optimizer, criterion, num_points, in_channels, device):
    model.train() # Set the model to training mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, data in enumerate(train_loader):
        x, y = data.x.to(device), data.y.to(device) # Move data to device

        x = x.view(-1, num_points, in_channels) # Reshape to (batch_size, num_points, in_channels)

        optimizer.zero_grad() # Zero the gradients
        logits = model(x) # Forward pass
        loss = criterion(logits, y) # Calculate loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights

        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total_samples += y.size(0)
        correct_predictions += (predicted == y).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    print(f'Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy