import numpy as np

def generate_audio(model, seed_audio, num_samples_to_generate, sequence_length, quantization_channels):
    generated_audio = list(seed_audio.flatten().astype(np.int32)) # Start with seed as list of integers

    print(f"Starting audio generation. Generating {num_samples_to_generate} samples...")

    for i in range(num_samples_to_generate):
        # Take the most recent 'sequence_length' samples as input context
        # Pad with zeros if generated_audio is shorter than sequence_length
        current_context = generated_audio[-sequence_length:]
        if len(current_context) < sequence_length:
            padding = [0] * (sequence_length - len(current_context))
            current_context = padding + current_context

        # Reshape for model input: (batch, sequence_length, 1)
        model_input = tf.constant(np.array(current_context).reshape(1, sequence_length, 1), dtype=tf.float32)

        # Get model predictions (probability distribution for each timestep)
        predictions = model(model_input)

        # We only care about the prediction for the *last* timestep to predict the *next* sample
        next_sample_probs = predictions[0, -1, :].numpy() # shape (quantization_channels,)

        # Sample from the probability distribution (categorical sampling)
        next_sample = np.random.choice(quantization_channels, p=next_sample_probs)

        generated_audio.append(next_sample)

        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{num_samples_to_generate} samples.")

    return np.array(generated_audio[len(seed_audio):], dtype=np.int32) # Return only newly generated samples