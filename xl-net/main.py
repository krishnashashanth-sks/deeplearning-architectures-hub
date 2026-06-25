from model import XLNetModel,PermutationLanguageModel
from train import train
from inference import *

config = {
    'vocab_size': 32000,  # Example: size of the vocabulary, commonly 32k for XLNet
    'd_model': 768,      # Hidden size or dimensionality of the model
    'n_head': 12,        # Number of attention heads
    'd_inner': 3072,     # Inner dimensionality of the feed-forward networks (4 * d_model)
    'n_layer': 12,       # Number of Transformer-XL layers
    'dropout': 0.1,      # General dropout rate
    'dropatt': 0.1,      # Dropout rate for attention weights
    'mem_len': 512,      # Length of the memory for Transformer-XL
    'max_seq_len': 512   # Maximum sequence length for input
}

model = XLNetModel(config)
plm_head = PermutationLanguageModel(config)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
plm_head.train()

num_epochs = 5

# --- Start of fix: Define dummy data for training loop ---
batch_size = 2
seq_len = config['max_seq_len']
vocab_size = config['vocab_size']
num_layers = config['n_layer']

# Dummy input_ids (batch_size, seq_len)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

# Dummy attention_mask (batch_size, seq_len)
# 1 for valid tokens, 0 for padding. Let's assume no padding for simplicity.
attention_mask = torch.ones(batch_size, seq_len).bool()

# Permutation mask for attention (batch_size, seq_len, seq_len)
# This is for the XLNetModel's attention mechanism to define which tokens can attend to which in a permutation.
attn_perm_mask = torch.rand(batch_size, seq_len, seq_len) < 0.1 # Mask ~10% of positions for attention
attn_perm_mask = attn_perm_mask.bool()

# Prediction mask for the PLM head (batch_size, seq_len)
# This specifies which tokens in the sequence are targeted for prediction for the PLM loss.
# For simplicity, let's randomly select 15% of tokens to be predicted.
plm_prediction_mask = torch.rand(batch_size, seq_len) < 0.15
plm_prediction_mask = plm_prediction_mask.bool()

# Dummy target_ids (batch_size, seq_len)
# Ground truth for masked tokens. Values don't matter much for a dummy run, as long as shape is correct.
target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

# Initialize memories as an empty list for the first iteration
mems = [None] * num_layers # Each layer has its own memory

# --- End of fix ---

train(num_epochs,optimizer,model,plm_head)

# Set models to evaluation mode
model.eval()
plm_head.eval()

print("--- Example 1: Predicting a masked token ---")
inference_seq_len = 10
input_ids_for_inference = torch.randint(0, config['vocab_size'], (1, inference_seq_len))
index_to_predict = 5
original_token_id_at_index = input_ids_for_inference[0, index_to_predict].item()

print(f"Input sequence IDs (sample): {input_ids_for_inference.tolist()[0]}")
print(f"Original token ID at index {index_to_predict}: {original_token_id_at_index}")

predicted_token_id = predict_masked_token(model, plm_head, config, input_ids_for_inference, index_to_predict)

print(f"Predicted token ID at index {index_to_predict}: {predicted_token_id}")
print(f"Was prediction correct? {predicted_token_id == original_token_id_at_index}")


print("\n--- Example 2: Autoregressive text generation ---")
start_token_id = 101 # An arbitrary token ID, e.g., for [CLS]
num_generate_tokens = 5

generated_token_ids = generate_text(model, plm_head, config, start_token_id, num_generate_tokens)

print(f"Generated sequence of {num_generate_tokens} tokens: {generated_token_ids}")