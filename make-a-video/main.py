import torch
from model import MakeAVideoPipeline

# --- Example Usage for the Advanced MakeAVideoPipeline ---
if __name__ == '__main__':
    print("\n--- Advanced MakeAVideoPipeline ---")

    # Configuration parameters
    vae_pixel_height = 64
    vae_pixel_width = 64

    pipeline_params = {
        'vocab_size': 10000,
        'text_embedding_dim': 768,
        'text_hidden_dim': 3072,
        'text_num_attention_heads': 12,
        'text_num_layers': 6,
        'vae_in_channels': 3,
        'vae_img_height': vae_pixel_height,
        'vae_img_width': vae_pixel_width,
        'vae_latent_dim': 128, # Match VAEEncoder final linear layer output
        'latent_channels': 4, # Output channels of VAE encoder's conceptual latent space
        'num_frames': 8, # Shorter video for quicker run
        'latent_height': vae_pixel_height // 8, # Assuming 3 downsamples in VAE encoder
        'latent_width': vae_pixel_width // 8, # Assuming 3 downsamples in VAE encoder
        'motion_embedding_dim': 256,
        'appearance_embedding_dim': 128,
        'unet_model_channels': 32,
        'unet_num_heads': 8,
        'unet_num_res_blocks': 1,
        'unet_channel_mults': (1, 2),
        'num_diffusion_steps': 20, # Reduced steps for quick example
        'diffusion_schedule_type': 'linear',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # Instantiate the pipeline
    pipeline = MakeAVideoPipeline(**pipeline_params)

    # Simulate input
    mock_text_prompts = [
        "a dog running in a park",
        "a bird flying over mountains"
    ]
    mock_initial_pixel_image = torch.randn(
        len(mock_text_prompts), pipeline_params['vae_in_channels'],
        pipeline_params['vae_img_height'], pipeline_params['vae_img_width'],
        device=pipeline_params['device']
    )

    print(f"Input text prompts: {mock_text_prompts}")
    print(f"Input initial pixel image shape: {mock_initial_pixel_image.shape}")

    # Generate video
    generated_video = pipeline.generate_video(
        text_prompts=mock_text_prompts,
        tokenizer=None, # Conceptual tokenizer
        initial_pixel_image=mock_initial_pixel_image,
        num_inference_steps=10 # Use fewer inference steps for faster example
    )

    print(f"\nGenerated pixel-space video shape: {generated_video.shape}")
    expected_output_shape = (
        len(mock_text_prompts),
        pipeline_params['num_frames'],
        pipeline_params['vae_in_channels'],
        pipeline_params['vae_img_height'],
        pipeline_params['vae_img_width']
    )
    assert generated_video.shape == expected_output_shape
    print("MakeAVideoPipeline successfully generated video with expected shape!")

    # Test without initial image
    print("\n--- MakeAVideoPipeline Example (without initial image) ---")
    generated_video_no_image = pipeline.generate_video(
        text_prompts=["a fish swimming in a coral reef"],
        tokenizer=None,
        initial_pixel_image=None,
        num_inference_steps=5
    )
    print(f"\nGenerated pixel-space video (no image) shape: {generated_video_no_image.shape}")
    expected_output_shape_no_image = (
        1,
        pipeline_params['num_frames'],
        pipeline_params['vae_in_channels'],
        pipeline_params['vae_img_height'],
        pipeline_params['vae_img_width']
    )
    assert generated_video_no_image.shape == expected_output_shape_no_image
    print("MakeAVideoPipeline successfully generated video without initial image!")
