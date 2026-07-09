import torch

def train_model(num_epochs,model,dataloader,optimizer,criterion,word_to_idx,vocabulary_size):
    # Set model to training mode
    model.train()

    print("Starting training with re-instantiated model...")

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # Assuming batch is a tensor of token IDs
            inputs = batch.long() # Ensure input is LongTensor for embedding

            # Prepare targets: shifted input for next word prediction
            # For input [w1, w2, ..., wn], target is [w2, w3, ..., wn, <pad>]
            # We need a target for each input token, so the last token's target is <pad>
            # The <pad> token ID is 0
            targets = torch.cat((inputs[:, 1:], torch.full((inputs.size(0), 1), word_to_idx["<pad>"], dtype=torch.long)), dim=1)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Reshape for CrossEntropyLoss
            # outputs: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            # targets: (batch_size, seq_len) -> (batch_size * seq_len)
            loss = criterion(outputs.view(-1, vocabulary_size), targets.view(-1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 1 == 0: # Print every batch for small dataset
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} finished, Average Loss: {total_loss / len(dataloader):.4f}\n")

    print("Training complete.")