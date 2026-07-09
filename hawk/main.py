import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import HawkModel
from train import train_model
from evaluate import evaluate_model

# Define example parameters 
vocab_size = 100
input_dim = 10
hidden_dim = 20
num_layers = 3
output_dim = vocab_size # For next token prediction
max_seq_len = 50
batch_size = 4
seq_len = 5 # Use a smaller seq_len for dummy input

# Create a dummy input tensor for token IDs
dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

# Instantiate the HawkModel
hawk_model = HawkModel(
    vocab_size=vocab_size,
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    output_dim=output_dim,
    max_seq_len=max_seq_len
)

# Pass the dummy input through the model
output = hawk_model(dummy_input_ids)

print(f"Dummy input_ids shape: {dummy_input_ids.shape}")
print(f"HawkModel output shape: {output.shape}")

# Basic assertion to check the output shape
assert output.shape == (batch_size, seq_len, output_dim), \
    f"Expected output shape ({batch_size}, {seq_len}, {output_dim}), but got {output.shape}"
print("HawkModel forward pass test successful. Output shape matches expected.")

# Define parameters for the dummy dataset
num_samples = 1000  # Number of sequences
seq_len = 50      # Length of each sequence (compatible with HawkModel max_seq_len)
vocab_size = 100  # Number of unique tokens (compatible with HawkModel vocab_size)
batch_size = 32   # Batch size for DataLoader

# Generate dummy input IDs
dummy_input_ids = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)

# Generate dummy target IDs (for next token prediction or similar tasks)
dummy_target_ids = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)

# Create a TensorDataset
dummy_dataset = TensorDataset(dummy_input_ids, dummy_target_ids)

# Create a DataLoader
dummy_dataloader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

print(f"Dummy dataset generated with {num_samples} samples, sequence length {seq_len}, and vocab size {vocab_size}.")
print(f"DataLoader created with batch size {batch_size}.")

loss_function = nn.CrossEntropyLoss()

learning_rate = 1e-4
optimizer = optim.AdamW(hawk_model.parameters(), lr=learning_rate)

# Define the number of training epochs
num_epochs = 5

#  Set up the model to run on a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hawk_model.to(device)
print(f"HawkModel moved to device: {device}")

train_model(num_epochs,hawk_model,dummy_dataloader,optimizer,loss_function,vocab_size,device)

print(f"Loss function (CrossEntropyLoss) and optimizer (AdamW with lr={learning_rate}) initialized.")
# Define parameters for the dummy validation dataset
# Using a smaller number of samples for validation
num_validation_samples = 200

# Generate dummy input IDs for validation
# Ensure different random samples than training data
dummy_validation_input_ids = torch.randint(0, vocab_size, (num_validation_samples, seq_len), dtype=torch.long)

# Generate dummy target IDs for validation
dummy_validation_target_ids = torch.randint(0, vocab_size, (num_validation_samples, seq_len), dtype=torch.long)

# Create a TensorDataset for validation
dummy_validation_dataset = TensorDataset(dummy_validation_input_ids, dummy_validation_target_ids)

# Create a DataLoader for validation (shuffle=False for consistent evaluation)
dummy_validation_dataloader = DataLoader(dummy_validation_dataset, batch_size=batch_size, shuffle=False)

print(f"Dummy validation dataset generated with {num_validation_samples} samples, sequence length {seq_len}, and vocab size {vocab_size}.")
print(f"Validation DataLoader created with batch size {batch_size}.")

#  Set the hawk_model to evaluation mode
hawk_model.eval()
print("HawkModel set to evaluation mode.")
avg_val_loss,accuracy=evaluate_model(dummy_validation_dataloader,hawk_model,loss_function,vocab_size,dummy_validation_dataset,device)

#  Print the average validation loss and accuracy
print(f"Validation complete. Average Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

# Ensure the model is in evaluation mode (already set from previous step, but good practice)
hawk_model.eval()
print("HawkModel is in evaluation mode.")

#  Prepare New Input
# Let's create a single dummy sequence for inference
inference_batch_size = 1
inference_seq_len = 5 # Can be different from training seq_len, but should be <= max_seq_len
dummy_inference_input_ids = torch.randint(0, vocab_size, (inference_batch_size, inference_seq_len), dtype=torch.long)
print(f"Dummy inference input shape: {dummy_inference_input_ids.shape}")

#  Move Input to Device
inference_input_ids = dummy_inference_input_ids.to(device)

#  Perform Forward Pass within torch.no_grad()
print("Performing inference...")
with torch.no_grad():
    logits = hawk_model(inference_input_ids)

#  Interpret Output
# Get predicted token IDs
predicted_ids = torch.argmax(logits, dim=-1) # (inference_batch_size, inference_seq_len)

print(f"Inference output (logits) shape: {logits.shape}")
print(f"Predicted token IDs shape: {predicted_ids.shape}")

# Display input and predicted output
print("\n--- Inference Results ---")
print("Input IDs:")
print(dummy_inference_input_ids)
print("\nPredicted IDs:")
print(predicted_ids)

# Example of what the logits look like for the first token in the sequence:
print("\nLogits for the first token in the first sequence:")
print(logits[0, 0, :])