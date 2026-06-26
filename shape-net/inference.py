import tensorflow as tf

def infer_advanced_model(encoder_model, decoder_model, num_samples,num_points_example,input_feature_dim_example):
    print("\nStarting TensorFlow model inference...")
    # Generate dummy input data using TensorFlow, matching expected dimensions
    # Assuming input_feature_dim_example and num_points_example are globally defined
    input_pc_batch = tf.random.normal((num_samples, num_points_example, input_feature_dim_example), dtype=tf.float32)

    print(f"Inferring {num_samples} samples...")

    # Encode the input point cloud
    latent_vectors = encoder_model(input_pc_batch, training=False) # training=False for inference
    print(f"Latent vectors shape: {latent_vectors.shape}")

    # Decode the latent vectors to reconstruct point clouds
    reconstructed_pcs = decoder_model(latent_vectors, training=False) # training=False for inference
    print(f"Reconstructed point clouds shape: {reconstructed_pcs.shape}")

    print("Inference complete. Displaying first reconstructed point cloud sample:\n", reconstructed_pcs[0, :5, :].numpy())

    return reconstructed_pcs
