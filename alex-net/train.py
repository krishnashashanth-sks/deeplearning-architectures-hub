import torch

def train(num_epochs,model,train_loader,test_loader,optimizer,criterion,device):
    # Initialize empty lists to store training and validation loss, and training and validation accuracy over epochs.
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print("Starting model training...")

    # Loop through each epoch:
    for epoch in range(num_epochs):
        # a. Set the model to training mode
        model.train()
        # b. Initialize variables to track running training loss and correct predictions.
        running_train_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        # c. Iterate through the train_loader:
        for i, data in enumerate(train_loader, 0):
            # i. Get inputs and labels, and move them to the device.
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # ii. Zero the gradients using optimizer.zero_grad().
            optimizer.zero_grad()

            # iii. Perform a forward pass: outputs = model(inputs).
            outputs = model(inputs)

            # iv. Calculate the loss: loss = criterion(outputs, labels).
            loss = criterion(outputs, labels)

            # v. Perform a backward pass: loss.backward().
            loss.backward()

            # vi. Update model parameters: optimizer.step().
            optimizer.step()

            # vii. Update running training loss and correct predictions.
            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train_samples += labels.size(0)
            correct_train_predictions += (predicted == labels).sum().item()

        # d. Calculate average training loss and training accuracy for the epoch.
        avg_train_loss = running_train_loss / total_train_samples
        train_accuracy = 100 * correct_train_predictions / total_train_samples

        # e. Set the model to evaluation mode using model.eval().
        model.eval()
        # f. Initialize variables to track running validation loss and correct predictions.
        running_val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0

        # g. Disable gradient calculations using torch.no_grad():
        with torch.no_grad():
            # i. Iterate through the test_loader:
            for data in test_loader:
                # 1. Get inputs and labels, and move them to the device.
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # 2. Perform a forward pass: outputs = model(inputs).
                outputs = model(inputs)

                # 3. Calculate the loss: loss = criterion(outputs, labels).
                loss = criterion(outputs, labels)

                # 4. Update running validation loss and correct predictions.
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                correct_val_predictions += (predicted == labels).sum().item()

        # h. Calculate average validation loss and validation accuracy for the epoch.
        avg_val_loss = running_val_loss / total_val_samples
        val_accuracy = 100 * correct_val_predictions / total_val_samples

        # i. Print the training and validation loss and accuracy for the current epoch.
        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
            f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

        # j. Append the training and validation loss and accuracy to their respective lists.
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
    return train_losses,train_accuracies,val_losses,val_accuracies