from model import Tacotron2
from config import HParams
from torch.utils.data import  DataLoader
from dataset import DummyDataset
from utils import collate_fn
import torch
from visualize import *
import torch.optim as optim
from train import train_step

# Instantiate HParams
hparams = HParams()

# Create a dummy batch of data for demonstration
batch_size = 4
max_text_len = 50
max_mel_len = 200 # max_mel_len / n_frames_per_step = max_decoder_steps

text_inputs = torch.randint(0, hparams.n_symbols, (batch_size, max_text_len), dtype=torch.long)
text_lengths = torch.randint(10, max_text_len, (batch_size,), dtype=torch.long)

mel_targets = torch.randn(batch_size, hparams.n_mel_channels, max_mel_len)
mel_lengths = torch.randint(50, max_mel_len, (batch_size,), dtype=torch.long)

# Instantiate the Tacotron2 model
model = Tacotron2(hparams)
print(f"Tacotron2 model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

# --- Forward pass (training) ---
# In training, mel_targets and mel_lengths are provided
mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model(
    text_inputs, text_lengths, mel_targets, mel_lengths)


# --- Inference pass ---
# In inference, only text_inputs and text_lengths are provided
print("\n--- Inference example ---")
inf_mel_outputs, inf_mel_outputs_postnet, inf_gate_outputs, inf_alignments = model.inference(
    text_inputs, text_lengths)


# Instantiate the Tacotron2 model
model = Tacotron2(hparams)
print(f"Tacotron2 model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

# Instantiate a dummy dataset
dummy_dataset = DummyDataset(
    num_samples=10,
    n_symbols=hparams.n_symbols,
    n_mel_channels=hparams.n_mel_channels,
    min_text_len=10,
    max_text_len=50,
    min_mel_len=50,
    max_mel_len=200
)

# Create a DataLoader with the custom collate_fn
dummy_dataloader = DataLoader(
    dummy_dataset,
    batch_size=4,
    shuffle=False, # Shuffle in real training, but not for this demo
    collate_fn=lambda batch: collate_fn(batch, hparams.n_frames_per_step)
)

mel_criterion = torch.nn.L1Loss()
gate_criterion = torch.nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

epochs = 5 # Small number for demonstration


print("Starting training loop...")
for epoch in range(epochs):
    total_epoch_loss = 0.0
    total_mel_loss = 0.0
    total_mel_postnet_loss = 0.0
    total_gate_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    for i, batch in enumerate(dummy_dataloader):
        num_batches += 1
        loss, mel_loss, mel_postnet_loss, gate_loss, grad_norm = train_step(model, optimizer,mel_criterion,gate_criterion, batch, hparams)

        total_epoch_loss += loss
        total_mel_loss += mel_loss
        total_mel_postnet_loss += mel_postnet_loss
        total_gate_loss += gate_loss
        total_grad_norm += grad_norm

    avg_epoch_loss = total_epoch_loss / num_batches
    avg_mel_loss = total_mel_loss / num_batches
    avg_mel_postnet_loss = total_mel_postnet_loss / num_batches
    avg_gate_loss = total_gate_loss / num_batches
    avg_grad_norm = total_grad_norm / num_batches

    print(f"Epoch {epoch+1}/{epochs} | Total Loss: {avg_epoch_loss:.4f} | Mel Loss: {avg_mel_loss:.4f} | Postnet Mel Loss: {avg_mel_postnet_loss:.4f} | Gate Loss: {avg_gate_loss:.4f} | Grad Norm: {avg_grad_norm:.4f}")

print("Training loop finished.")

# Select the first sample from the inference outputs for visualization
# Move to CPU for plotting if on GPU
sample_idx = 0

inference_mel_output = inf_mel_outputs_postnet[sample_idx]
inference_alignment = inf_alignments[sample_idx]

# Plot the generated mel-spectrogram
plot_mel_spectrogram(inference_mel_output, title=f"Generated Mel-spectrogram (Sample {sample_idx+1})")

# Plot the attention alignment
plot_attention_alignment(inference_alignment, title=f"Attention Alignment (Sample {sample_idx+1})")
