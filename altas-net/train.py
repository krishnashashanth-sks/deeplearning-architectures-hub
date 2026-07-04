import torch
from utils import calculate_accuracy

def train_model(num_epochs,model,train_loader,val_loader,optimizer,criterion,scheduler,device):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0
    checkpoint_interval = 10

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += calculate_accuracy(outputs, labels) * inputs.size(0)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        scheduler.step()

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += calculate_accuracy(outputs, labels) * inputs.size(0)
                val_total_samples += inputs.size(0)

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects / val_total_samples
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc
            torch.save(model.state_dict(), 'model_best_accuracy.pth')
            print(f"Saved best model with validation accuracy: {best_val_accuracy:.4f}")

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), f'model_checkpoint_epoch_{epoch+1}.pth')
            print(f"Saved model checkpoint at epoch {epoch+1}")

    print("\nTraining complete!")
    return train_losses,val_losses,train_accuracies,val_accuracies 