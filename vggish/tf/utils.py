import tensorflow as tf

# Architectural constants.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96     # with zero overlap.

# Parameters used for embedding postprocessing.
PCA_EIGEN_VECTORS_NAME = 'pca_eigen_vectors'
PCA_MEANS_NAME = 'pca_means'
QUANTIZE_MIN_VAL = -2.0
QUANTIZE_MAX_VAL = +2.0


def waveform_to_log_mel_spectrograms_tf(audio_path, sr=SAMPLE_RATE):
    """
    Loads an audio file and converts it to VGGish-compatible log-mel spectrograms
    using only TensorFlow operations.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Target sample rate. VGGish uses 16000 Hz.

    Returns:
        tf.Tensor: A tensor of shape (num_frames, num_mel_bands, 1) representing
                   the log-mel spectrograms, or None if an error occurs.
    """
    try:
        # 1. Load audio using tf.io.read_file and tf.audio.decode_wav
        audio_binary = tf.io.read_file(audio_path)
        waveform, original_sr = tf.audio.decode_wav(audio_binary, desired_channels=1) # Decode to mono
        waveform = tf.squeeze(waveform, axis=-1) # Remove channel dimension if mono

        # 2. Resample if necessary
        if original_sr != sr:
            waveform = tf.audio.resample(waveform, original_sr, sr) # TensorFlow 2.x resample

        # 3. Define STFT parameters (from vggish_params)
        stft_window_length_samples = int(round(STFT_WINDOW_LENGTH_SECONDS * sr))
        stft_hop_length_samples = int(round(STFT_HOP_LENGTH_SECONDS * sr))
        fft_size = int(round(sr * STFT_WINDOW_LENGTH_SECONDS))

        # 4. Compute Short-Time Fourier Transform (STFT)
        stfts = tf.signal.stft(
            signals=waveform,
            frame_length=stft_window_length_samples,
            frame_step=stft_hop_length_samples,
            fft_length=fft_size,
            window_fn=tf.signal.hann_window, # VGGish uses Hann window
            pad_end=False # Don't pad frames if not enough samples
        )

        magnitude_spectrograms = tf.abs(stfts)

        # 5. Build Mel Spectrogram
        num_spectrogram_bins = stfts.shape[-1]
        linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=NUM_MEL_BINS, # Corrected from NUM_MEL_BANDS
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sr,
            lower_edge_hertz=MEL_MIN_HZ,
            upper_edge_hertz=MEL_MAX_HZ
        )
        mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_matrix, 1)
        mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
            linear_to_mel_matrix.shape[-1:]))

        # 6. Apply Log transformation and adjust dynamic range
        log_offset = LOG_OFFSET
        log_mel_spectrograms = tf.math.log(mel_spectrograms + log_offset)

        # VGGish divides audio into 0.96s examples, each 96 frames tall and 64 mel bins wide
        # Frame length of 0.025s and hop length of 0.010s means (0.96 - 0.025)/0.010 + 1 = 94.5, so 96 frames (rounding up)
        # It's more complex with padding and framing; for simplicity here, we'll return the full spec for now.
        # The original VGGish preprocesses and then stacks 96 frames.

        # Convert to VGGish example batches (96 frames x 64 mel bins)
        # This part requires careful framing of the log-mel spectrograms
        examples_per_second = sr / EXAMPLE_HOP_SECONDS # Corrected from EXAMPLE_HOP_LENGTH_SECONDS
        example_window_frames = int(round(EXAMPLE_WINDOW_SECONDS / STFT_HOP_LENGTH_SECONDS))
        example_hop_frames = int(round(EXAMPLE_HOP_SECONDS / STFT_HOP_LENGTH_SECONDS))

        # Pad the spectrogram if it's too short for even one example frame
        total_frames = tf.shape(log_mel_spectrograms)[0]
        if total_frames < example_window_frames:
            padding = example_window_frames - total_frames
            log_mel_spectrograms = tf.pad(log_mel_spectrograms, [[0, padding], [0, 0]])
            total_frames = tf.shape(log_mel_spectrograms)[0]

        # Use tf.signal.frame to create example batches
        frames_to_stack = tf.signal.frame(
            log_mel_spectrograms,
            frame_length=example_window_frames, # 96 frames
            frame_step=example_hop_frames,     # 96 frames (overlap for continuous examples)
            pad_end=False,                     # No extra padding if not enough frames
            axis=0
        )

        # Each frame should be (96, 64). Add a channel dimension for Keras Conv2D input
        examples_batch = tf.expand_dims(frames_to_stack, axis=-1) # Shape: (num_examples, 96, 64, 1)

        return examples_batch

    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None