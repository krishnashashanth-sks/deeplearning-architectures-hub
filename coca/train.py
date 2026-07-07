def train_model(epochs,contrastive_captioner,dataset,ckpt_manager):
    # --- Training Loop ---
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # Reset the metrics at the start of the next epoch
        contrastive_captioner.loss_tracker.reset_state()

        for (batch, (img_batch, cap_batch)) in enumerate(dataset):
            loss_metrics = contrastive_captioner.train_step((img_batch, cap_batch))
            total_loss += loss_metrics['loss'].numpy()
            num_batches += 1

            if batch % 10 == 0: # Log every 10 batches
                print(f'Epoch {epoch+1} Batch {batch} Loss {loss_metrics["loss"].numpy():.4f}')

        # Calculate average loss for the epoch
        avg_epoch_loss = total_loss / num_batches
        print(f'\nEpoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}')

        # Save checkpoint every epoch
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print("\nTraining complete!")