def train(num_epochs,optimizer,model,plm_head):
    print("Starting training loop...")

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass through XLNetModel
        # Pass mems as a list of tensors
        h_out, g_out, new_mems = model(input_ids, attention_mask=attention_mask, mems=mems, perm_mask=attn_perm_mask)

        # Calculate loss using PermutationLanguageModel
        # Only compute loss for positions where plm_prediction_mask is True
        loss, logits = plm_head(g_out, target_ids, plm_prediction_mask)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update memories for the next iteration
        mems = new_mems

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("Training loop finished.")