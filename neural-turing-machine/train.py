import torch

def train_model(num_epochs,ntm_model,training_data,loss_function,optimizer,batch_size,device):
    print("Starting training...")

    for epoch in range(num_epochs):
        ntm_model.train() # Set model to training mode
        total_loss = 0
        for batch_idx, (input_seq, target_seq) in enumerate(training_data):
            ntm_model.reset(batch_size, device)
            optimizer.zero_grad()
            output_seq = ntm_model(input_seq)

            output_seq_length = output_seq.shape[1]
            sliced_target_seq = target_seq[:, :output_seq_length, :]

            loss = loss_function(output_seq, sliced_target_seq)
            loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ntm_model.parameters(), max_norm=10)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Training finished.")