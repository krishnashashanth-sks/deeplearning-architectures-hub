import torch

def train_model(num_epochs,model,trainloader,testloader,optimizer,criterion,scheduler,device):
    train_losses,train_accuracies=[],[]
    val_losses,val_accuracies=[],[]
    for epoch in range(num_epochs):
        # Training phase
        model.train() # Set model to training mode
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        scheduler.step()

        avg_train_loss = running_train_loss / len(trainloader)
        train_accuracy = 100. * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad(): # Disable gradient calculation for validation
            for batch_idx, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        avg_val_loss = running_val_loss / len(testloader)
        val_accuracy = 100. * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}]\t ' \
            f'Train Loss: {avg_train_loss:.4f}\t ' \
            f'Train Acc: {train_accuracy:.2f}%\t ' \
            f'Val Loss: {avg_val_loss:.4f}\t ' \
            f'Val Acc: {val_accuracy:.2f}%')

    print("Training complete.")
    return train_losses,train_accuracies,val_losses,val_accuracies