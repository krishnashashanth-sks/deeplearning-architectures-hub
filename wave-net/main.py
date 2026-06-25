from model import WaveNet
from utils import *
from train import train_step
from inference import generate_audio
import soundfile as sf
import numpy as np

wavenet_model = WaveNet(
    num_blocks=2, # Number of block cycles
    num_layers_per_block=3, # Dilation rates will be 1, 2, 4 per block cycle
    filters=32, # Filters for the gated activation convolutions
    kernel_size=2,
    residual_filters=128, # Filters for residual path, changed to match skip_filters
    skip_filters=128, # Filters for skip path
    output_dim=256 # For 8-bit audio (256 possible values)
)

print(wavenet_model.summary())

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Parameters for audio
sample_rate = 16000  # Hz
duration = 1         # seconds
freq = 440           # A4 note
quantization_channels = wavenet_model.output_dim # Use the model's output_dim (256)

# Generate a synthetic audio signal (e.g., a sine wave)
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
synthetic_audio = 0.5 * np.sin(2 * np.pi * freq * t) # Scale to [-0.5, 0.5]

print(f"Original synthetic audio shape: {synthetic_audio.shape}")
print(f"Original synthetic audio min/max: {np.min(synthetic_audio):.4f} / {np.max(synthetic_audio):.4f}")

# Apply mu-law encoding
encoded_audio = mu_law_encode(synthetic_audio, quantization_channels)

print(f"Mu-law encoded audio shape: {encoded_audio.shape}")
print(f"Mu-law encoded audio min/max: {np.min(encoded_audio)} / {np.max(encoded_audio)}")
print(f"Mu-law encoded audio data type: {encoded_audio.dtype}")

# Prepare input and target sequences for WaveNet
# Input: all samples except the last one
# Target: all samples except the first one (shifted by one timestep)

x_train = encoded_audio[:-1]
y_train = encoded_audio[1:]

# Reshape for the model: (batch, sequence_length, 1) for input
x_train = np.expand_dims(x_train, axis=0) # Add batch dimension
x_train = np.expand_dims(x_train, axis=-1) # Add feature dimension

y_train = np.expand_dims(y_train, axis=0) # Add batch dimension

num_epochs = 5

print(f"\nStarting training for {num_epochs} epochs using synthetic audio data...")

for epoch in range(num_epochs):
    # Use the preprocessed synthetic audio data
    loss = train_step(wavenet_model, tf.constant(x_train, dtype=tf.float32), tf.constant(y_train, dtype=tf.int32),loss_fn,optimizer)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy():.4f}")

print("Training finished.")

# Parameters for generation
# Use a small segment of the preprocessed synthetic audio as a seed
seed_length = 50 # How many samples from the existing data to use as a starting point
seed_audio_segment = x_train[0, :seed_length, 0] # Take first 'seed_length' samples from x_train (batch 0, feature 0)

# Ensure seed_audio_segment is 1D
if seed_audio_segment.ndim > 1:
    seed_audio_segment = seed_audio_segment.flatten()

# Let's generate 5 seconds of audio at 16000 Hz sample rate
desired_duration_seconds = 5
num_samples_to_generate = desired_duration_seconds * sample_rate # Generate 5 seconds of audio

# Perform inference
generated_encoded_audio = generate_audio(
    wavenet_model,
    seed_audio_segment,
    num_samples_to_generate,
    sequence_length=100, # Use a sequence_length for context, consistent with training setup
    quantization_channels=output_dim
)

print(f"\nGenerated encoded audio shape: {generated_encoded_audio.shape}")
print(f"Generated encoded audio min/max: {np.min(generated_encoded_audio)} / {np.max(generated_encoded_audio)}")

# Decode the generated mu-law audio
decoded_generated_audio = mu_law_decode(generated_encoded_audio, output_dim)

print(f"Decoded generated audio shape: {decoded_generated_audio.shape}")
print(f"Decoded generated audio min/max: {np.min(decoded_generated_audio):.4f} / {np.max(decoded_generated_audio):.4f}")


output_filename = "generated_wavenet_audio.wav"
sf.write(output_filename, decoded_generated_audio, sample_rate)
print(f"Audio saved to {output_filename}")