def train(model, device, train_loader, criterion,optimizer, epoch):
    model.train() # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # Clear gradients
        output = model(data) # Forward pass
        loss = criterion(output, target) # Calculate loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights

        if batch_idx % 100 == 0: # Print training status every 100 batches
            # Calculate accuracy for the current batch
            pred = output.argmax(dim=1, keepdim=True) # Get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / len(data)
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')