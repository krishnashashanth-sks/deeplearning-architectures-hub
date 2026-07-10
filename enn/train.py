# --- Training Function ---
def train(model, device, train_loader, optimizer, epoch,criterion):
    model.train() # Set the model to training mode
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0: # Print every 100 batches
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_train_loss = running_loss / len(train_loader)
    print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}')

