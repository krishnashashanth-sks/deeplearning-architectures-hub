import torch
import numpy as np

# ---  Sampling Function ---
def sample_images(encoder, unet_model, decoder, device, num_samples=8, latent_channels=4, image_size=32):
    encoder.eval()
    unet_model.eval()
    decoder.eval()

    # Calculate latent spatial dimensions based on encoder structure (e.g., two stride=2 downsamples)
    # Assuming Encoder has 2 downsample blocks for 32x32 -> 8x8 latent
    latent_h = image_size // (2**2)
    latent_w = image_size // (2**2)
    
    # Generate random noise as a starting point for the latent space
    # In a real LCM, this would be a specific sampling trajectory
    noise = torch.randn(num_samples, latent_channels, latent_h, latent_w).to(device)
    
    # Simulate a single timestep for the UNet (e.g., 0 for initial state)
    timesteps_for_sampling = torch.zeros(num_samples, dtype=torch.long, device=device) 

    with torch.no_grad():
        # Pass through UNet (consistency function) and then decoder
        refined_latent = unet_model(noise, timesteps_for_sampling)
        generated_images = decoder(refined_latent)

        generated_images = (generated_images + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
        generated_images = torch.clamp(generated_images, 0.0, 1.0) # Clip to ensure valid pixel range
        generated_images = generated_images.cpu().numpy()
        generated_images = np.transpose(generated_images, (0, 2, 3, 1)) # (N, C, H, W) to (N, H, W, C)

    return generated_images