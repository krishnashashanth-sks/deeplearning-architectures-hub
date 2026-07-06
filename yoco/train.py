import torch

def train_model(epochs,model,train_dataloader,val_dataloader,optimizer,yoco_loss_fn,scheduler,log_interval,device):
    print("Starting training loop...")

    for epoch in range(epochs):
        model.train() # Set model to training mode
        total_loss_epoch = 0
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = yoco_loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()

            if batch_idx % log_interval == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}")

        scheduler.step()
        print(f"Epoch {epoch+1} finished. Average Train Loss: {total_loss_epoch / len(train_dataloader):.4f}. Current LR: {optimizer.param_groups[0]['lr']}")

        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for val_batch_idx, (images, targets) in enumerate(val_dataloader):
                images = images.to(device)
                targets = targets.to(device)

                predictions = model(images)
                val_loss = yoco_loss_fn(predictions, targets)
                val_loss_epoch += val_loss.item()

        print(f"Validation Loss after Epoch {epoch+1}: {val_loss_epoch / len(val_dataloader):.4f}")

    print("Training complete.")