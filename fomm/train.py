# Training loop
def train_model(fom_model,num_epochs,optimizer,reconstruction_loss_fn,dataloader,device):
    fom_model.train() # Set the model to training mode

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (source_image, driving_image) in enumerate(dataloader):
            source_image = source_image.to(device)
            driving_image = driving_image.to(device)

            optimizer.zero_grad() # Clear gradients

            # Forward pass
            output = fom_model(source_image, driving_image)
            generated_image = output['generated_image']

            # Calculate loss (here, a simple reconstruction loss between generated and driving image)
            # For a full FOMM, you would have more sophisticated losses and potentially ground truth driving video frames
            loss = reconstruction_loss_fn(generated_image, driving_image)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Training complete!")
