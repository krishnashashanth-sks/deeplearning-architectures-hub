import torch
import math

def train(epochs,model,train_dataloader,val_dataloader,loss_fn,optimizer,device):
    for epoch in range(epochs):
        model.train() # Set the model to training mode
        total_loss = 0
        for batch_idx, (xb, yb) in enumerate(train_dataloader):
            # Move batch to device
            xb, yb = xb.to(device), yb.to(device)

            # Forward pass
            logits, _ = model(xb, mode='parallel')

            # Reshape for CrossEntropyLoss: (N, C) and (N)
            # logits: (batch_size, block_size, vocab_size) -> (batch_size * block_size, vocab_size)
            # yb: (batch_size, block_size) -> (batch_size * block_size)
            logits = logits.view(-1, logits.size(-1))
            yb = yb.view(-1)

            # Calculate loss
            loss = loss_fn(logits, yb)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 1000 == 0: # Print loss every 1000 batches
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} finished, Average Training Loss: {avg_loss:.4f}")    
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad(): # Disable gradient calculation for evaluation
            for xb, yb in val_dataloader:
                xb, yb = xb.to(device), yb.to(device)

                logits, _ = model(xb, mode='parallel')

                # Reshape for CrossEntropyLoss
                logits = logits.view(-1, logits.size(-1))
                yb = yb.view(-1)

                loss = loss_fn(logits, yb)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        perplexity = math.exp(avg_val_loss) # Perplexity is exp(average_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}")