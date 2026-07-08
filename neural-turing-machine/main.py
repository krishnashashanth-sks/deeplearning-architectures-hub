import torch
import torch.nn as nn
import torch.optim as optim
from model import NTM
from utils import generate_copy_sequence
from train import train_model

# --- Parameters and Initialization ---

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
batch_size = 1
min_seq_len = 1
max_seq_len = 10
vector_size = 8
num_batches = 100

# Delimiter vectors
start_delimiter = torch.zeros(1, vector_size)
start_delimiter[0, 0] = 1.0

end_delimiter = torch.zeros(1, vector_size)
end_delimiter[0, 1] = 1.0

# Generate training data
training_data = []
for _ in range(num_batches):
    input_seq, target_seq = generate_copy_sequence(
        batch_size, min_seq_len, max_seq_len, vector_size, start_delimiter, end_delimiter
    )
    training_data.append((input_seq.to(device), target_seq.to(device)))

print(f"Generated {num_batches} batches of data for training.")

# NTM model parameters
input_size = vector_size
output_size = vector_size
mem_size = 128
mem_vector_size = vector_size
num_read_heads = 1
num_write_heads = 1
controller_hidden_size = 100

ntm_model = NTM(input_size, output_size, mem_size, mem_vector_size,
                num_read_heads, num_write_heads, controller_hidden_size).to(device)

loss_function = nn.BCEWithLogitsLoss(reduction='none')
learning_rate = 0.001
optimizer = optim.Adam(ntm_model.parameters(), lr=learning_rate)

# --- Training Loop ---
num_epochs = 50

train_model(num_epochs,ntm_model,training_data,loss_function,optimizer,batch_size,device)

# --- Inference --- 
ntm_model.eval() # Set model to evaluation mode

# Generate a new random sequence for inference
inf_batch_size = 1
inf_min_seq_len = 5
inf_max_seq_len = 8

input_inference, target_inference = generate_copy_sequence(
    inf_batch_size, inf_min_seq_len, inf_max_seq_len, vector_size, start_delimiter, end_delimiter
)
input_inference = input_inference.to(device)
target_inference = target_inference.to(device)

print(f"Inference Input Sequence (shape {input_inference.shape}):\n{input_inference.int()}")

with torch.no_grad():
    ntm_model.reset(inf_batch_size, device)
    output_inference_logits = ntm_model(input_inference)
    output_inference = torch.sigmoid(output_inference_logits) # Apply sigmoid to get probabilities

# For binary prediction, threshold at 0.5
predicted_output = (output_inference > 0.5).float()

print(f"\nInference Target Sequence (shape {target_inference.shape}):\n{target_inference.int()}")
print(f"\nInference Predicted Output (shape {predicted_output.shape}):\n{predicted_output.int()}")

# Compare a portion of the target and predicted output for clarity
output_seq_length_inference = output_inference.shape[1]
sliced_target_inference = target_inference[:, :output_seq_length_inference, :]

# Focus on the data part of the output (after the input sequence has been processed)
# The input sequence length is input_inference.shape[1]
# The expected data output starts right after the input sequence finishes.
start_data_output_idx = input_inference.shape[1]

if start_data_output_idx < predicted_output.shape[1]:
    print(f"\nComparing Data Copy (Target vs Predicted) from index {start_data_output_idx} onwards:")
    print("Target data part:")
    print(sliced_target_inference[:, start_data_output_idx:, 2:].int())
    print("\nPredicted data part:")
    print(predicted_output[:, start_data_output_idx:, 2:].int())

    accuracy = (predicted_output[:, start_data_output_idx:, 2:] == sliced_target_inference[:, start_data_output_idx:, 2:]).float().mean().item()
    print(f"\nData Copy Accuracy: {accuracy*100:.2f}%")
else:
    print("\nPredicted output sequence is too short to compare data copy.")