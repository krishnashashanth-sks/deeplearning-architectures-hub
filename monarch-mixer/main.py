import torch.nn as nn
import torch.optim as optim
from model import M2Model
import torch

# 1. Define the following parameters:
vocab_size = 10000
sequence_length = 128
batch_size = 32
num_classes = 10

print(f"Vocab Size: {vocab_size}")
print(f"Sequence Length: {sequence_length}")
print(f"Batch Size: {batch_size}")
print(f"Number of Classes: {num_classes}")

# 2. Generate a dummy input tensor x (token IDs)
x = torch.randint(0, vocab_size, (batch_size, sequence_length))

# 3. Generate a dummy target label tensor y
y = torch.randint(0, num_classes, (batch_size,))

print(f"\nShape of dummy input x: {x.shape}")
print(f"Sample of dummy input x:\n{x[0, :5]}") # Print first 5 tokens of the first sample
print(f"\nShape of dummy target y: {y.shape}")
print(f"Sample of dummy target y:\n{y[:5]}") # Print first 5 target labels

# 1. Define appropriate values for model parameters
embed_dim = 64
num_layers = 2
kernel_size = 3
num_blocks_conv = 4
intermediate_dim_conv = 16
num_blocks_mlp = 4
intermediate_dim_mlp = 16
dropout_rate = 0.1

print(f"Embedding Dimension: {embed_dim}")
print(f"Number of M2 Blocks: {num_layers}")
print(f"Kernel Size for Conv: {kernel_size}")
print(f"Number of Blocks for Conv: {num_blocks_conv}")
print(f"Intermediate Dimension for Conv Blocks: {intermediate_dim_conv}")
print(f"Number of Blocks for MLP: {num_blocks_mlp}")
print(f"Intermediate Dimension for MLP Blocks: {intermediate_dim_mlp}")
print(f"Dropout Rate: {dropout_rate}")

# 2. Instantiate the M2Model
model = M2Model(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_layers=num_layers,
    kernel_size=kernel_size,
    num_blocks_conv=num_blocks_conv,
    intermediate_dim_conv=intermediate_dim_conv,
    num_blocks_mlp=num_blocks_mlp,
    intermediate_dim_mlp=intermediate_dim_mlp,
    dropout_rate=dropout_rate,
    num_classes=num_classes
)

# 3. Define a loss function
criterion = nn.CrossEntropyLoss()

# 4. Define an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Print the model summary and the total number of trainable parameters
print("\n--- M2Model Architecture ---")
print(model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params}")

# 1. Define the number of epochs for training
num_epochs = 5

print(f"Starting training for {num_epochs} epochs...")

# Move model to device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
x = x.to(device)
y = y.to(device)

# 2. Start a loop that iterates num_epochs times
for epoch in range(num_epochs):
    # 3. Inside the epoch loop, set the model to training mode
    model.train()

    # 4. Perform a forward pass
    outputs = model(x)

    # 5. Calculate the loss
    loss = criterion(outputs, y)

    # 6. Zero out the gradients of the optimizer
    optimizer.zero_grad()

    # 7. Perform backpropagation
    loss.backward()

    # 8. Update the model's parameters
    optimizer.step()

    # 9. Print the epoch number and the calculated loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training loop finished.")

# 1. Generate new dummy input data x_val and target labels y_val for a validation set

x_val = torch.randint(0, vocab_size, (batch_size, sequence_length))
y_val = torch.randint(0, num_classes, (batch_size,))

print(f"\nShape of dummy validation input x_val: {x_val.shape}")
print(f"Shape of dummy validation target y_val: {y_val.shape}")

# 2. Move x_val and y_val to the appropriate device
x_val = x_val.to(device)
y_val = y_val.to(device)


# 3. Set the model to evaluation mode
model.eval()

# 4. Disable gradient calculations for the evaluation phase
with torch.no_grad():
    # 5. Perform a forward pass with the x_val data
    val_outputs = model(x_val)

    # 6. Calculate the predictions
    predictions = torch.argmax(val_outputs, dim=1)

    # 7. Calculate the accuracy
    correct_predictions = (predictions == y_val).sum().item()
    accuracy = correct_predictions / batch_size

    # 8. Print the calculated validation accuracy
    print(f"Validation Accuracy: {accuracy:.4f}")
