import torch
import torch.optim as optim
from model import JambaModel
import torch.nn as nn

#1.Model parameters
d_model=128
num_heads=4,
d_ff=512,
vocab_size = 10000  # Example vocabulary size
num_layers = 2      # Example number of Jamba layers
max_seq_len = 10 # Use the same seqlen for max_seq_len
d_state = 16        # Example value for SSM
d_conv = 3          # Example value for SSM
expand = 2    
batch_size = 4
num_experts=8
learning_rate=1e-4
top_k=2

# 2. Generate dummy input_ids tensor
# `input_ids` should contain integers representing token IDs.
dummy_input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len))

# 3. Generate dummy target_labels tensor
# `target_labels` should also contain integers representing the ground truth token IDs for next token prediction.
dummy_target_labels = torch.randint(0, vocab_size, (batch_size, max_seq_len))

# 4. Generate dummy attention_mask tensor
# For simplicity, starting with all ones, indicating no padding.
dummy_attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool)

# 5. Print the shapes of the generated tensors
print(f"Shape of dummy_input_ids: {dummy_input_ids.shape}")
print(f"Shape of dummy_target_labels: {dummy_target_labels.shape}")
print(f"Shape of dummy_attention_mask: {dummy_attention_mask.shape}")

# 1. Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Re-instantiate JambaModel to ensure it uses the latest class definition
model = JambaModel(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    d_state=d_state,
    d_conv=d_conv,
    expand=expand,
    num_experts=num_experts,
    top_k=top_k,
    max_seq_len=max_seq_len
)

# Move model to device
model.to(device)

# Move dummy data to device
dummy_input_ids = dummy_input_ids.to(device)
dummy_target_labels = dummy_target_labels.to(device)
dummy_attention_mask = dummy_attention_mask.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

loss_function = nn.CrossEntropyLoss()

# Set model to training mode
model.train()

# Parameters for training
num_epochs = 10
alpha = 0.01 # Scaling factor for load balancing loss

print("Starting training...")
for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    logits, all_expert_loads, all_expert_probabilities = model(dummy_input_ids, attention_mask=dummy_attention_mask)

    # Calculate main loss
    # Reshape logits to (batch_size * seq_len, vocab_size)
    # Reshape target_labels to (batch_size * seq_len)
    main_loss = loss_function(logits.view(-1, vocab_size), dummy_target_labels.view(-1))

    # Calculate load balancing loss
    load_balancing_loss = 0.0
    for expert_load, expert_probability in zip(all_expert_loads, all_expert_probabilities):
        # Add a small epsilon to prevent log(0) if any probability is exactly zero
        # This might not be strictly necessary with current calculation, but good practice
        # Also ensure `expert_load` and `expert_probability` are on the same device
        load_balancing_loss += (expert_probability.to(device) * expert_load.to(device)).sum()

    # Total loss
    total_loss = main_loss + alpha * load_balancing_loss

    # Backward pass
    total_loss.backward()

    # Optimizer step
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss.item():.4f}, Main Loss: {main_loss.item():.4f}, LB Loss: {load_balancing_loss.item():.4f}")

print("Training loop finished.")

# 1. Generate new dummy_inference_input_ids
dummy_inference_input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len)).to(device)

# 2. Generate new dummy_inference_attention_mask
dummy_inference_attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool).to(device)

print(f"Dummy inference input IDs shape: {dummy_inference_input_ids.shape}")
print(f"Dummy inference attention mask shape: {dummy_inference_attention_mask.shape}")

print("Setting model to evaluation mode and performing inference...")
# 3. Set the model to evaluation mode
model.eval()

# 4. Perform a forward pass within torch.no_grad()
with torch.no_grad():
    # The JambaModel forward method now returns logits, all_expert_loads, and all_expert_probabilities
    # For inference, we are primarily interested in the logits.
    inference_logits, _, _ = model(dummy_inference_input_ids, attention_mask=dummy_inference_attention_mask)

# 5. Print the shape of the logits
print(f"Inference Logits shape: {inference_logits.shape}")

# Optional: Set model back to train mode if further training is expected
model.train()

# 1. Take the inference_logits tensor
# inference_logits is already available from the previous step

# 2. For each token position, apply argmax() along the vocab_size dimension
predicted_token_ids = torch.argmax(inference_logits, dim=-1)

# 3. Print the shape and a small sample of the predicted_token_ids
print(f"Shape of predicted_token_ids: {predicted_token_ids.shape}")
print("Sample predicted_token_ids (first batch, first few tokens):")
print(predicted_token_ids[0, :5])
