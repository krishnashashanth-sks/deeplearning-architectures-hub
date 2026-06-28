import torch

def train(num_epochs,lnn_model,train_dataloader,val_dataloader,dummy_t,loss_function,optimizer,device):
    print("\n--- Starting Training and Evaluation Loop ---")

    for epoch in range(num_epochs):
        lnn_model.train() # Set model to training mode
        total_train_loss = 0

        for h0_batch, x_seq_batch, target_batch in train_dataloader:
            # Move data to the correct device
            h0_batch = h0_batch.to(device)
            x_seq_batch = x_seq_batch.to(device)
            target_batch = target_batch.to(device)

            # dummy_t is a global tensor, already on device

            optimizer.zero_grad()

            # Forward pass
            predictions = lnn_model(h0_batch, x_seq_batch, dummy_t)

            # Calculate loss
            loss = loss_function(predictions, target_batch)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

        # Evaluation loop
        lnn_model.eval() # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad(): # Disable gradient calculation for evaluation
            for h0_batch, x_seq_batch, target_batch in val_dataloader:
                # Move data to the correct device
                h0_batch = h0_batch.to(device)
                x_seq_batch = x_seq_batch.to(device)
                target_batch = target_batch.to(device)

                predictions = lnn_model(h0_batch, x_seq_batch, dummy_t)
                val_loss = loss_function(predictions, target_batch)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

    print("--- Training and Evaluation Complete ---")