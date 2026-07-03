from snntorch import functional as SF

def train_snn(model, train_loader, optimizer, loss_fn, num_epochs, num_steps, device):
    loss_hist,acc_hist=[],[]
    print("Starting SNN training...")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device) # Move input data to device
            targets = targets.to(device) # Move target labels to device

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # Model expects flattened input for the first layer, so data.flatten(1) is needed
            spk_rec = model(data, num_steps) # Output spikes are recorded over num_steps

            # Calculate loss (snnTorch.functional.ce_rate_loss expects output spikes and targets)
            loss = loss_fn(spk_rec, targets)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy for the current batch
            # snnTorch's accuracy_rate expects output spikes and targets
            batch_acc = SF.accuracy_rate(spk_rec, targets)
            correct_predictions += batch_acc * len(targets)
            total_samples += len(targets)

        avg_loss = total_loss / len(train_loader)
        epoch_accuracy = (correct_predictions / total_samples) * 100

        loss_hist.append(avg_loss)
        acc_hist.append(epoch_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")
    print("SNN training finished.")
    return loss_hist,acc_hist