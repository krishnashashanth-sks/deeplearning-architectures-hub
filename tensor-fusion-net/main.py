import torch.optim as optim
import torch
import torch.nn as nn
from model import TFNModel

# Hyperparameters for dummy dataset and model

# Dataset parameters
batch_size = 2
sequence_length = 20 # For text and audio sequences
vocab_size = 1000 # For text embeddings
embed_dim = 64 # Text embedding dimension
audio_feature_dim = 32 # Number of audio features per time step
image_height = 16 # Visual input height
image_width = 16 # Visual input width
visual_channels = 3 # RGB channels for visual input
num_classes = 3 # For classification task (e.g., positive, neutral, negative)

# Feature extractor parameters
text_hidden_dim = 32
audio_hidden_dim = 16
visual_hidden_dim = 24

# Common convolutional parameters for feature extractors
num_filters_text = 50
kernel_sizes_text = [3, 4, 5]

num_filters_audio = 32
kernel_sizes_audio = [2, 3, 4]

num_filters_visual = 16
kernel_sizes_visual = [3, 5] # Using 'same' padding for conv2d

# Prediction head parameters
task_type = 'classification' # or 'regression'

# ---  Instantiate the TFN Model ---
model = TFNModel(
    vocab_size=vocab_size, embed_dim=embed_dim, text_hidden_dim=text_hidden_dim,
    num_filters_text=num_filters_text, kernel_sizes_text=kernel_sizes_text,
    audio_feature_dim=audio_feature_dim, audio_hidden_dim=audio_hidden_dim,
    num_filters_audio=num_filters_audio, kernel_sizes_audio=kernel_sizes_audio,
    visual_channels=visual_channels, visual_hidden_dim=visual_hidden_dim,
    num_filters_visual=num_filters_visual, kernel_sizes_visual=kernel_sizes_visual,
    image_height=image_height, image_width=image_width,
    num_classes=num_classes, task_type=task_type
)

print(model)

# ---  Create Dummy Input Tensors ---
# Text input: (batch_size, sequence_length)
dummy_text_input = torch.randint(0, vocab_size, (batch_size, sequence_length))
print(f"Dummy Text Input Shape: {dummy_text_input.shape}")

# Audio input: (batch_size, audio_feature_dim, sequence_length)
# For Conv1d, channels typically come first, so (batch_size, channels, sequence_length)
dummy_audio_input = torch.randn(batch_size, audio_feature_dim, sequence_length)
print(f"Dummy Audio Input Shape: {dummy_audio_input.shape}")

# Visual input: (batch_size, visual_channels, image_height, image_width)
# For Conv2d, channels typically come first
dummy_visual_input = torch.randn(batch_size, visual_channels, image_height, image_width)
print(f"Dummy Visual Input Shape: {dummy_visual_input.shape}")

# Dummy target tensor (for classification)
dummy_target = torch.randint(0, num_classes, (batch_size,))
print(f"Dummy Target Shape: {dummy_target.shape}")

# ---  Define Loss Function and Optimizer ---

# Loss function
if task_type == 'classification':
    criterion = nn.CrossEntropyLoss()
else: # regression
    criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Loss Function: {criterion.__class__.__name__}")
print(f"Optimizer: {optimizer.__class__.__name__}")

# ---  Implement Basic Training Loop ---

num_epochs = 5 # Small number of epochs for demonstration

print("\nStarting basic training loop...")
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    # Ensure dummy inputs are passed in the correct order as defined in TFNModel's forward method
    outputs = model(dummy_text_input, dummy_audio_input, dummy_visual_input)

    # Calculate loss
    # For classification, target should be class indices (long type)
    loss = criterion(outputs, dummy_target)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Basic training loop finished.")

# 1. Set the instantiated model to evaluation mode
model.eval()
print("Model set to evaluation mode.")

# 2. Create new dummy text input tensors for testing/inference
dummy_test_text_input = torch.randint(0, vocab_size, (batch_size, sequence_length))

# 3. Create new dummy audio input tensors for testing/inference
dummy_test_audio_input = torch.randn(batch_size, audio_feature_dim, sequence_length)

# 4. Create new dummy visual input tensors for testing/inference
dummy_test_visual_input = torch.randn(batch_size, visual_channels, image_height, image_width)

# 1. Use torch.no_grad() for inference
with torch.no_grad():
    # 2. Pass the dummy test inputs through the model
    predictions = model(dummy_test_text_input, dummy_test_audio_input, dummy_test_visual_input)

# 3. Print the shape of the obtained predictions
print(f"Predictions Shape: {predictions.shape}")

# 4. Print a sample of the predictions
print("Sample Predictions:")
print(predictions)

# 1. Determine the predicted class for each sample
# For classification with softmax output, the predicted class is the one with the highest probability
predicted_classes = torch.argmax(predictions, dim=1)

# 2. Compare these predicted classes with the dummy_target tensor to find the number of correct predictions
correct_predictions = (predicted_classes == dummy_target).sum().item()

# 3. Calculate the accuracy
total_samples = dummy_target.size(0)
accuracy = correct_predictions / total_samples

# 4. Print the predicted classes, the actual target classes, and the calculated accuracy
print(f"\nPredicted Classes: {predicted_classes}")
print(f"Actual Target Classes: {dummy_target}")
print(f"Number of Correct Predictions: {correct_predictions}/{total_samples}")
print(f"Accuracy: {accuracy:.4f}")