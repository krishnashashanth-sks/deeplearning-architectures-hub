# --- Training Loop Definition ---
def train_model(model, dataloader, optimizer, loss_fn, num_epochs=10):
    model.train() # Set the model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (rdk_feats, maccs_feats, ecfp_feats, protein_feats, affinities) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass
            predictions = model(rdk_feats, maccs_feats, ecfp_feats, protein_feats)

            # Calculate loss
            loss = loss_fn(predictions, affinities)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")

    print("Training complete!")