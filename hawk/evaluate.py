import torch

def evaluate_model(validation_dataloader,hawk_model,loss_function,vocab_size,dummy_validation_dataset,device):
    total_val_loss = 0.0
    total_correct_predictions = 0
    total_tokens = 0

    print("Starting validation loop...")
    with torch.no_grad():
        # 4. Iterate through batches in the validation_dataloader
        for input_ids, target_ids in validation_dataloader:
            # 5. Move input_ids and target_ids to the appropriate device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # 6. Perform a forward pass
            logits = hawk_model(input_ids) # (batch_size, seq_len, vocab_size)

            # 7. Calculate the batch validation loss
            loss = loss_function(logits.view(-1, vocab_size), target_ids.view(-1))

            # 8. Accumulate the batch validation loss
            total_val_loss += loss.item() * input_ids.size(0) # Multiply by batch size for correct average

            # 9. Calculate batch accuracy
            # Get predicted token IDs by taking argmax of logits
            predicted_ids = torch.argmax(logits, dim=-1) # (batch_size, seq_len)

            # Compare predictions with target_ids
            correct_predictions = (predicted_ids == target_ids).sum().item()
            total_correct_predictions += correct_predictions

            total_tokens += target_ids.numel() # Total elements in the target tensor

    # 10. Calculate the average validation loss and overall accuracy for the epoch
    avg_val_loss = total_val_loss / len(dummy_validation_dataset)
    accuracy = total_correct_predictions / total_tokens
    return avg_val_loss,accuracy