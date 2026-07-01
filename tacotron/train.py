import torch
from utils import get_mask_from_lengths

def train_step(model, optimizer,mel_criterion,gate_criterion, batch, hparams):
    model.train()
    texts_padded, text_lengths, mels_padded, gate_padded, mel_lengths = batch

    # Move data to GPU if available
    if torch.cuda.is_available():
        texts_padded = texts_padded.cuda()
        text_lengths = text_lengths.cuda()
        mels_padded = mels_padded.cuda()
        gate_padded = gate_padded.cuda()
        mel_lengths = mel_lengths.cuda()

    optimizer.zero_grad()

    # Forward pass
    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model(
        texts_padded, text_lengths, mels_padded, mel_lengths)

    # Calculate Mel Loss
    # The mel_lengths mask is important to only compute loss on actual mel frames
    mask = ~get_mask_from_lengths(mel_lengths, mels_padded.size(2))
    mask = mask.unsqueeze(1).expand_as(mel_outputs)

    mel_loss = mel_criterion(mel_outputs.masked_select(mask), mels_padded.masked_select(mask))
    mel_postnet_loss = mel_criterion(mel_outputs_postnet.masked_select(mask), mels_padded.masked_select(mask))

    # Calculate Gate Loss
    gate_loss = gate_criterion(gate_outputs.view(-1), gate_padded.view(-1))

    total_loss = mel_loss + mel_postnet_loss + gate_loss

    # Backward pass and optimization
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
    optimizer.step()

    return total_loss.item(), mel_loss.item(), mel_postnet_loss.item(), gate_loss.item(), grad_norm.item()
