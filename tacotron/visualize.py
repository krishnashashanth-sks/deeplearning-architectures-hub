import matplotlib.pyplot as plt
import seaborn as sns

def plot_mel_spectrogram(mel_spectrogram, title="Mel-spectrogram"):
    """Plots a mel-spectrogram."""
    plt.figure(figsize=(12, 4))
    sns.heatmap(mel_spectrogram.cpu().detach().numpy().T, cmap='magma', cbar=True, yticklabels=False)
    plt.title(title)
    plt.xlabel("Decoder Time Step")
    plt.ylabel("Mel Bin")
    plt.show()

def plot_attention_alignment(alignment, title="Attention Alignment"):
    """Plots an attention alignment matrix."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(alignment.cpu().detach().numpy().T, cmap='viridis', cbar=True)
    plt.title(title)
    plt.xlabel("Encoder Time Step")
    plt.ylabel("Decoder Time Step")
    plt.show()
