# Training Function
def train_moco(model, dataloader, optimizer, loss_fn, num_epochs, device):
    """Performs the MoCo v2 training loop."""
    print(f"\nStarting MoCo v2 training for {num_epochs} epochs on {device}...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images_q, images_k) in enumerate(dataloader):
            images_q = images_q.to(device)
            images_k = images_k.to(device)

            logits, labels = model(images_q, images_k)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0: # Log every 10 batches
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
    print("\nMoCo v2 training complete.")