import torch

def test(model, test_loader, criterion,num_points, in_channels, device):
    model.eval() # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # Disable gradient calculation during inference
        for data in test_loader:
            x, y = data.x.to(device), data.y.to(device)

            # Reshape x similar to the train function
            x = x.view(-1, num_points, in_channels) # Reshape to (batch_size, num_points, in_channels)

            logits = model(x) # Forward pass
            loss = criterion(logits, y) # Calculate loss

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_samples += y.size(0)
            correct_predictions += (predicted == y).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy