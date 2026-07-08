import torch
import torch.nn as nn
import torch.optim as optim
from model import SCNNModel

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Retrieve variables from kernel state
nside_initial = 16
in_channels = 1
num_classes = 10
batch_size = 4
num_pixels = 12 * nside_initial**2 

model = SCNNModel(nside_initial=nside_initial, in_channels=in_channels, num_classes=num_classes)
model.to(device)

loss_function = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("Loss function and optimizer re-initialized.")

# Prepare dummy data and move to the selected device
# These variables were already created in the previous step but need to be moved to the device.
x = torch.randn(batch_size, in_channels, num_pixels, dtype=torch.float32).to(device)
y = torch.randint(0, num_classes, (batch_size,), dtype=torch.long).to(device)

print(f"Dummy input x moved to {device}, shape: {x.shape}")
print(f"Dummy target y moved to {device}, shape: {y.shape}")

# Define number of training epochs
num_epochs = 5 # Let's train for 5 epochs for demonstration

# Training Loop
model.train() # Set model to training mode
print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    optimizer.zero_grad() # Zero the gradients for this batch

    # Forward pass
    outputs = model(x)

    # Calculate loss
    loss = loss_function(outputs, y)

    # Backward pass
    loss.backward()

    # Update model parameters
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training loop completed.")

# 1. Define a test_batch_size
test_batch_size = 4 # Using the same batch_size as training for consistency in this dummy example

# 2. Create a dummy input tensor x_test
x_test = torch.randn(test_batch_size, in_channels, num_pixels, dtype=torch.float32)

# 3. Create a dummy target label tensor y_test
y_test = torch.randint(0, num_classes, (test_batch_size,), dtype=torch.long)

# 4. Move both x_test and y_test to the device
x_test = x_test.to(device)
y_test = y_test.to(device)

print(f"Dummy test input x_test created and moved to {x_test.device}, shape: {x_test.shape}, Dtype: {x_test.dtype}")
print(f"Dummy test target y_test created and moved to {y_test.device}, shape: {y_test.shape}, Dtype: {y_test.dtype}")

# 1. Set the model to evaluation mode
model.eval()

# 2. Disable gradient calculations for evaluation
with torch.no_grad():
    # 3. Perform a forward pass on the x_test data
    outputs_test = model(x_test)

    # 4. Calculate the evaluation loss
    eval_loss = loss_function(outputs_test, y_test)

    # 5. Convert the model's raw outputs (logits) into predicted class labels
    # The outputs are (batch_size, num_classes). argmax along dim=1 gives the predicted class.
    _, predicted = torch.max(outputs_test.data, 1)

    # 6. Calculate the accuracy
    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()
    accuracy = 100 * correct / total

    # 7. Print the calculated evaluation loss and accuracy
    print(f"Evaluation Loss: {eval_loss.item():.4f}")
    print(f"Evaluation Accuracy: {accuracy:.2f}%")

# 1. Create a new dummy input tensor x_inference
# Using the same batch_size as x_test for consistency
inference_batch_size = 4 # Using the same as test_batch_size
x_inference = torch.randn(inference_batch_size, in_channels, num_pixels, dtype=torch.float32)
print(f"Dummy inference input x_inference created, shape: {x_inference.shape}, Dtype: {x_inference.dtype}")

# 2. Move the x_inference tensor to the computational device
x_inference = x_inference.to(device)
print(f"Dummy inference input x_inference moved to {x_inference.device}.")

# 3. Set the model to evaluation mode
model.eval()
print("Model set to evaluation mode for inference.")

# 4. Perform a forward pass on x_inference to obtain the model's outputs (logits)
# Ensure that gradient calculations are disabled
with torch.no_grad():
    outputs_inference = model(x_inference)
    print(f"Inference outputs shape (logits): {outputs_inference.shape}")

    # 5. Convert the model's output logits into predicted class labels
    # The outputs are (batch_size, num_classes). argmax along dim=1 gives the predicted class.
    _, predicted_labels = torch.max(outputs_inference.data, 1)

# 6. Print the predicted class labels for the x_inference data
print("\nPredicted class labels for inference data:")
print(predicted_labels)
