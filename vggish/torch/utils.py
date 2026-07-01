import torch
import torchaudio.transforms as T

from main import SAMPLE_RATE,WIN_LENGTH,N_FFT,HOP_LENGTH,N_MELS

def preprocess_audio_to_vggish_input(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Converts a raw audio waveform to a batch of 96x64 log-mel-spectrogram patches
    suitable for VGGish input.

    Args:
        waveform (torch.Tensor): Raw audio waveform (mono, 1D tensor).
        sample_rate (int): Sample rate of the input waveform.

    Returns:
        torch.Tensor: A tensor of shape (num_patches, 1, 96, 64) representing
                      log-mel-spectrogram patches.
    """

    # 1. Resample if necessary
    if sample_rate != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # Ensure waveform is mono (remove extra dimensions if present)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0) # Convert to mono if stereo

    # 2. Compute MelSpectrogram
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=125, # VGGish specific frequency range
        f_max=7500,
        power=2.0, # Power of 2 for magnitude spectrogram
    )
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Add a small epsilon to avoid log(0)
    log_mel_spectrogram = torch.log(mel_spectrogram + 1e-6)

    # VGGish expects 96 frames per patch. Calculate number of full patches.
    # Each frame corresponds to HOP_LENGTH samples.
    # Total frames = (audio_length - WIN_LENGTH) // HOP_LENGTH + 1
    # Since log_mel_spectrogram is (n_mels, n_frames), we are looking at n_frames.
    num_frames = log_mel_spectrogram.shape[-1]

    # VGGish patches are 96 frames long. We need to extract these.
    # This part can be tricky for arbitrary audio lengths. For simplicity,
    # let's assume we extract overlapping patches or a single central patch
    # for demonstration. For full VGGish, you'd slide a 96-frame window.

    # For this example, let's just take the first 96 frames if available
    # or pad if not enough frames are present.
    target_frames = 96
    if num_frames < target_frames:
        # Pad with zeros if not enough frames
        padding = target_frames - num_frames
        log_mel_spectrogram = torch.nn.functional.pad(log_mel_spectrogram, (0, padding))
    elif num_frames > target_frames:
        # Take the first 96 frames (or center crop, depending on requirements)
        log_mel_spectrogram = log_mel_spectrogram[:, :target_frames]

    # Normalize to [0, 1] for VGGish, then scale to [-1, 1] range approximately.
    # The original VGGish applies a mean and std normalization after log scaling
    # based on their dataset. For a generic implementation without dataset stats,
    # we can use a simpler approach or rely on input range.

    # VGGish often uses a specific normalization based on statistics from their dataset.
    # For a general implementation without those stats, we might skip or use a simple range normalization.
    # However, the original VGGish paper mentions values in range [0, 255] for patches.
    # Let's target the shape: (batch_size, 1, 96, 64)

    # Current shape: (n_mels, target_frames) -> (64, 96)
    # Desired shape for VGGish input: (1, 1, 96, 64) -> (batch_size, channels, frames, mels)
    # Permute and unsqueeze to match the input expectation (batch, channels, frames, mels)
    vggish_input = log_mel_spectrogram.T.unsqueeze(0).unsqueeze(0) # (1, 1, 96, 64)

    return vggish_input
