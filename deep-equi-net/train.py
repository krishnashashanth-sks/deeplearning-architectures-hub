from tqdm.auto import tqdm
import torch

def train_model(num_epochs,model,train_loader,val_loader,optimizer,criterion,device):
    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print("Starting training...")

    for epoch in tqdm(range(num_epochs)):
        # --- Training Phase ---
        model.train() # Set the model to training mode
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in tqdm(enumerate(train_loader)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # Zero the gradients

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # --- Validation Phase ---
        model.eval() # Set the model to evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad(): # Disable gradient calculations for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
            f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    print("Training complete.")
    return train_losses,val_losses,train_accuracies,val_accuracies