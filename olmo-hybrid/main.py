import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from model import OLMoHybridModel
from dataset import TextDataset
from utils import tokenize_corpus,numericalize_and_pad_sequences
from train import train_model
import torch

num_heads = 8
ff_dim = 256
num_transformer_layers = 2
num_rnn_layers = 2
dropout_rate = 0.1 # Using the default dropout rate as defined in TransformerBlock and OLMoHybridModel
batch_size = 4
seq_len = 10
embed_dim = 128

model = OLMoHybridModel(
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_layers=num_transformer_layers,
    num_rnn_layers=num_rnn_layers,
    dropout_rate=dropout_rate
)

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Never jump over the lazy dog quickly.",
    "The dog barks at the fox.",
    "A quick brown dog is better than a lazy fox."
]

tokenized_sentences = tokenize_corpus(corpus)

special_tokens = {"<pad>": 0, "<unk>": 1}

word_to_idx = special_tokens.copy()
idx_to_word = list(special_tokens.keys())

# Start assigning IDs from the next available integer after special tokens
next_idx = len(special_tokens)

for sentence_tokens in tokenized_sentences:
    for word in sentence_tokens:
        if word not in word_to_idx:
            word_to_idx[word] = next_idx
            idx_to_word.append(word)
            next_idx += 1

vocabulary_size = len(word_to_idx)

max_sequence_length = max(len(s) for s in tokenized_sentences) # Or a fixed value like 10, using dynamic length for now



numericalized_padded_sequences = numericalize_and_pad_sequences(
    tokenized_sentences, word_to_idx, max_sequence_length
)

dataset = TextDataset(numericalized_padded_sequences)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#  Define the loss function
criterion = nn.CrossEntropyLoss()

model = OLMoHybridModel(
    vocab_size=vocabulary_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_layers=num_transformer_layers,
    num_rnn_layers=num_rnn_layers,
    dropout_rate=dropout_rate
)

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


num_epochs = 5
train_model(num_epochs,model,dataloader,optimizer,criterion,word_to_idx,vocabulary_size)


# Set model to evaluation mode
model.eval()

print("Generating predictions...")

# Get one batch from the dataloader for inference
with torch.no_grad(): # Disable gradient calculation for inference
    for batch_idx, batch in enumerate(dataloader):
        sample_input = batch.long()
        break # We only need one batch

    # Forward pass
    sample_output_logits = model(sample_input)

    # Get predicted token IDs (greedy decoding)
    # The argmax is applied over the vocabulary dimension
    predicted_token_ids = torch.argmax(sample_output_logits, dim=-1)

# Decode the input and predicted sequences
def decode_sequence(sequence_ids, idx_to_word):
    return " ".join([idx_to_word[idx.item()] for idx in sequence_ids if idx.item() != word_to_idx["<pad>"]])

print("\nSample Input and Generated Predictions:")
for i in range(min(2, sample_input.size(0))): # Display first 2 samples
    input_sequence = decode_sequence(sample_input[i], idx_to_word)
    predicted_sequence = decode_sequence(predicted_token_ids[i], idx_to_word)
    print(f"  Input: {input_sequence}")
    print(f"  Predicted: {predicted_sequence}")
