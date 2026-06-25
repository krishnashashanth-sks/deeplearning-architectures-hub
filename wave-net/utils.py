import numpy as np

# Define mu-law companding functions
def mu_law_encode(audio, quantization_channels):
    mu = quantization_channels - 1
    # x = np.tanh(audio)
    # The paper uses: x = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(1 + mu)
    # However, librosa's implementation is slightly different, handling scaling more directly.
    # Let's use the standard definition for simplicity and clarity.
    # Ensure audio is in range [-1, 1]
    audio = audio / np.max(np.abs(audio))
    encoded = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(1 + mu)
    return ((encoded + 1) / 2 * mu).astype(np.int32)

def mu_law_decode(encoded_audio, quantization_channels):
    mu = quantization_channels - 1
    y = encoded_audio / mu * 2 - 1
    decoded = np.sign(y) * (1 / mu) * ((1 + mu)**np.abs(y) - 1)
    return decoded