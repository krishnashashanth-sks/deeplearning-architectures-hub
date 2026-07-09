def train_model(num_epochs,hawk_model,data_loader,optimizer,loss_function,vocab_size,device):
    # Training loop
    print("Starting training loop...")
    for epoch in range(num_epochs):
        # 4. Set the hawk_model to training mode
        hawk_model.train()

        # 5. Initialize a variable to accumulate the running loss
        running_loss = 0.0
        total_batches = 0

        # 6. Iterate through batches in the data_loader
        for batch_idx, (input_ids, target_ids) in enumerate(data_loader):
            # 7. For each batch, move the input_ids and target_ids tensors to the appropriate device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # 8. Zero out the gradients of the optimizer
            optimizer.zero_grad()

            # 9. Perform a forward pass
            logits = hawk_model(input_ids) # (batch_size, seq_len, vocab_size)

            # 10. Calculate the loss
            # CrossEntropyLoss expects (N, C) for logits and (N) for targets
            # Reshape logits to (batch_size * seq_len, vocab_size)
            # Reshape target_ids to (batch_size * seq_len)
            loss = loss_function(logits.view(-1, vocab_size), target_ids.view(-1))

            # 11. Perform backpropagation
            loss.backward()

            # 12. Update model parameters
            optimizer.step()

            # 13. Accumulate the batch loss
            running_loss += loss.item()
            total_batches += 1

        # 14. After iterating through all batches in an epoch, print the average loss for that epoch
        avg_loss = running_loss / total_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Training loop finished.")