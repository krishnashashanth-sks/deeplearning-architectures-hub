import torch

def collate_fn(batch, n_frames_per_step=1):
    """Pads sequences with zeros to create a batch for Tacotron2 training.

    Args:
        batch (list): List of tuples, where each tuple contains:
            - text (torch.Tensor): Tensor of character embeddings (1D).
            - mel (torch.Tensor): Tensor of mel-spectrogram (n_mel_channels, mel_length).
            - text_length (int): Length of the text sequence.
            - mel_length (int): Length of the mel-spectrogram sequence.
        n_frames_per_step (int): Number of frames predicted per decoder step.

    Returns:
        tuple: Padded tensors for text, input lengths, mel-spectrogram, gate targets,
               and output lengths.
    """
    # Sort batch by text length in descending order
    batch.sort(key=lambda x: x[2], reverse=True)

    texts, mels, text_lengths, mel_lengths = zip(*batch)

    # Pad text sequences
    max_text_len = max(text_lengths)
    texts_padded = torch.zeros((len(texts), max_text_len), dtype=torch.long)
    for i, text in enumerate(texts):
        texts_padded[i, :text.size(0)] = text

    # Pad mel-spectrograms
    max_mel_len = max(mel_lengths)
    # Ensure mel_length is a multiple of n_frames_per_step
    if max_mel_len % n_frames_per_step != 0:
        max_mel_len += n_frames_per_step - (max_mel_len % n_frames_per_step)

    n_mel_channels = mels[0].size(0)
    mels_padded = torch.zeros((len(mels), n_mel_channels, max_mel_len), dtype=torch.float)
    gate_padded = torch.zeros((len(mels), max_mel_len // n_frames_per_step), dtype=torch.float)
    for i, mel in enumerate(mels):
        mels_padded[i, :, :mel.size(1)] = mel
        # Create gate target: 1 at the end of the sequence, 0 otherwise
        gate_padded[i, mel_lengths[i] // n_frames_per_step - 1] = 1

    text_lengths = torch.tensor(text_lengths, dtype=torch.long)
    mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)

    return texts_padded, text_lengths, mels_padded, gate_padded, mel_lengths

def get_mask_from_lengths(lengths, max_len):
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return ~mask