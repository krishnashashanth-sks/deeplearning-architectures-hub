import torch

# ---  Training Function ---
def train_lcm(
    encoder, decoder, unet_model, discriminator, 
    optimizer_encoder, optimizer_decoder, optimizer_unet, optimizer_discriminator,
    trainloader, valloader, num_epochs, device,
    mse_loss, bce_loss, 
    lambda_reconstruction, lambda_consistency, lambda_adversarial,
    gradient_clip_value, log_interval
):
    best_val_loss = float('inf')
    print("Starting training loop...")

    # Generate some fixed noise for consistent sampling during training
    fixed_noise = torch.randn(8, 4, 32 // (2**2), 32 // (2**2)).to(device)
    fixed_timesteps = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70]).to(device) # Example timesteps

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        unet_model.train()
        if lambda_adversarial > 0: discriminator.train()

        running_total_g_loss = 0.0
        running_reconstruction_loss = 0.0
        running_consistency_loss = 0.0
        running_adversarial_g_loss = 0.0
        running_d_loss = 0.0

        for batch_idx, (images, _) in enumerate(trainloader):
            images = images.to(device)
            batch_size = images.shape[0]

            # --- Training Discriminator ---
            if lambda_adversarial > 0.0:
                optimizer_discriminator.zero_grad()
                
                with torch.no_grad(): # Generate fake images without affecting generator gradients
                    z_0 = encoder(images)
                    # Simulate time step for UNet, simplified for this example (e.g., random or fixed)
                    timesteps_unet = torch.randint(0, 64, (batch_size,), device=device)
                    # Noise for consistency - actual LCMs have specific noise schedules
                    noise = torch.randn_like(z_0) * 0.1 
                    z_t = z_0 + noise # Adding noise for UNet input
                    refined_latent = unet_model(z_t, timesteps_unet) # UNet processes noisy latent
                    fake_images = decoder(refined_latent).detach() # Detach for discriminator training

                real_output = discriminator(images)
                fake_output = discriminator(fake_images)

                d_loss_real = bce_loss(real_output, torch.ones_like(real_output))
                d_loss_fake = bce_loss(fake_output, torch.zeros_like(fake_output))
                d_loss = d_loss_real + d_loss_fake

                d_loss.backward()
                optimizer_discriminator.step()
                running_d_loss += d_loss.item()

            # --- Training Generator (Encoder, UNet, Decoder) ---
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            optimizer_unet.zero_grad()

            z_0 = encoder(images)
            
            # Reconstruction Loss
            reconstructed_images = decoder(z_0)
            L_reconstruction = mse_loss(reconstructed_images, images)

            # Consistency Loss (simplified: UNet aims to predict original latent from noisy latent)
            # In a true LCM, the UNet predicts the 'x_0' (original latent) given noisy 'x_t' and 't'.
            timesteps_unet = torch.randint(0, 64, (batch_size,), device=device)
            noise_for_consistency = torch.randn_like(z_0) * 0.1 # Simulate small noise for consistency
            z_t_consistent = z_0 + noise_for_consistency
            
            unet_predicted_z0 = unet_model(z_t_consistent, timesteps_unet) # UNet predicts clean latent
            L_consistency = mse_loss(unet_predicted_z0, z_0) # UNet should be consistent with clean latent

            total_generator_loss = lambda_reconstruction * L_reconstruction + \
                                   lambda_consistency * L_consistency

            L_adversarial_g = torch.tensor(0.0).to(device)
            if lambda_adversarial > 0.0:
                # Re-pass to get gradients for generator
                reconstructed_images_for_g_adv = decoder(refined_latent) # Use refined_latent from UNet
                gen_fake_output = discriminator(reconstructed_images_for_g_adv)
                L_adversarial_g = bce_loss(gen_fake_output, torch.ones_like(gen_fake_output))
                total_generator_loss += lambda_adversarial * L_adversarial_g
                running_adversarial_g_loss += L_adversarial_g.item()
            
            total_generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clip_value)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clip_value)
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), gradient_clip_value)
            optimizer_encoder.step()
            optimizer_decoder.step()
            optimizer_unet.step()

            # --- Logging ---
            running_total_g_loss += total_generator_loss.item()
            running_reconstruction_loss += L_reconstruction.item()
            running_consistency_loss += L_consistency.item()

            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(trainloader)}], "
                      f"G_Loss: {running_total_g_loss / (batch_idx + 1):.4f}, "
                      f"Rec_Loss: {running_reconstruction_loss / (batch_idx + 1):.4f}, "
                      f"Cons_Loss: {running_consistency_loss / (batch_idx + 1):.4f}" +
                      (f", Adv_G_Loss: {running_adversarial_g_loss / (batch_idx + 1):.4f}, "
                       f"D_Loss: {running_d_loss / (batch_idx + 1):.4f}" if lambda_adversarial > 0.0 else ""))

        # --- Validation Step (after each epoch) ---
        encoder.eval()
        decoder.eval()
        unet_model.eval()
        if lambda_adversarial > 0: discriminator.eval()

        val_total_g_loss = 0.0
        val_reconstruction_loss = 0.0
        val_consistency_loss = 0.0
        val_adversarial_g_loss = 0.0
        val_d_loss = 0.0

        with torch.no_grad():
            for val_batch_idx, (images, _) in enumerate(valloader):
                images = images.to(device)
                batch_size_val = images.shape[0]

                z_0 = encoder(images)
                reconstructed_images = decoder(z_0)
                L_reconstruction_val = mse_loss(reconstructed_images, images)

                timesteps_unet_val = torch.randint(0, 64, (batch_size_val,), device=device)
                noise_val = torch.randn_like(z_0) * 0.1
                z_t_val = z_0 + noise_val
                unet_predicted_z0_val = unet_model(z_t_val, timesteps_unet_val)
                L_consistency_val = mse_loss(unet_predicted_z0_val, z_0)

                total_generator_loss_val = lambda_reconstruction * L_reconstruction_val + \
                                           lambda_consistency * L_consistency_val

                if lambda_adversarial > 0.0:
                    # Re-run UNet and decoder to get images based on UNet output
                    refined_latent_val = unet_model(z_t_val, timesteps_unet_val) # UNet processes noisy latent
                    fake_images_val = decoder(refined_latent_val)

                    real_output_val = discriminator(images)
                    fake_output_val = discriminator(fake_images_val)

                    d_loss_real_val = bce_loss(real_output_val, torch.ones_like(real_output_val))
                    d_loss_fake_val = bce_loss(fake_output_val, torch.zeros_like(fake_output_val))
                    d_loss_val = d_loss_real_val + d_loss_fake_val
                    val_d_loss += d_loss_val.item()

                    gen_fake_output_val = discriminator(fake_images_val)
                    L_adversarial_g_val = bce_loss(gen_fake_output_val, torch.ones_like(gen_fake_output_val))
                    total_generator_loss_val += lambda_adversarial * L_adversarial_g_val
                    val_adversarial_g_loss += L_adversarial_g_val.item()

                val_total_g_loss += total_generator_loss_val.item()
                val_reconstruction_loss += L_reconstruction_val.item()
                val_consistency_loss += L_consistency_val.item()

        avg_val_g_loss = val_total_g_loss / len(valloader)
        avg_val_rec_loss = val_reconstruction_loss / len(valloader)
        avg_val_cons_loss = val_consistency_loss / len(valloader)

        print(f"\nValidation Epoch [{epoch+1}/{num_epochs}], "
              f"Avg G_Loss: {avg_val_g_loss:.4f}, "
              f"Avg Rec_Loss: {avg_val_rec_loss:.4f}, "
              f"Avg Cons_Loss: {avg_val_cons_loss:.4f}" +
              (f", Avg Adv_G_Loss: {val_adversarial_g_loss / len(valloader):.4f}, "
               f"Avg D_Loss: {val_d_loss / len(valloader):.4f}" if lambda_adversarial > 0.0 else ""))

        # --- Model Saving ---
        if avg_val_g_loss < best_val_loss:
            best_val_loss = avg_val_g_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'unet_state_dict': unet_model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict() if lambda_adversarial > 0.0 else None,
                'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
                'optimizer_unet_state_dict': optimizer_unet.state_dict(),
                'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict() if lambda_adversarial > 0.0 else None,
                'best_val_loss': best_val_loss,
            }, 'best_lcm_model_actual.pth')
            print(f"Saved best model with Validation G_Loss: {best_val_loss:.4f}\n")

    print("Training finished.")