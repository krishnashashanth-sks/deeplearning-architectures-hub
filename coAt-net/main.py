import matplotlib.pyplot as plt
from model import CoAtNet
import tensorflow as tf
import numpy as np

num_classes = 10

# Reduce parameters to avoid system crash
model = CoAtNet(
    num_classes=num_classes,
    input_shape=(224, 224, 3),
    num_blocks_per_stage=[1, 1, 2, 2, 1],  # Reduced number of blocks per stage
    filters_per_stage=[32, 64, 128, 256, 512], # Reduced filters per stage
    transformer_stages=[3, 4, 5], # Keep transformer stages, but with reduced params
    embed_dim_per_stage=[128, 256, 512], # Reduced embedding dimension
    num_heads_per_stage=[2, 4, 8], # Reduced number of attention heads
    ff_dim_per_stage=[256, 512, 1024] # Reduced feed-forward network dimension
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# Dummy data generation (replace with your actual dataset)
batch_size = 16 # Reduced batch size for lower memory usage
input_shape = (224, 224, 3)

dummy_x_train = np.random.rand(100, *input_shape).astype(np.float32) # 100 dummy images
dummy_y_train = -np.random.randint(0, num_classes, 100) # 100 dummy labels

dummy_x_val = np.random.rand(20, *input_shape).astype(np.float32) # 20 dummy images for validation
dummy_y_val = np.random.randint(0, num_classes, 20) # 20 dummy labels

print("\nStarting a short training run with dummy data...")

history = model.fit(
    x=dummy_x_train,  # Your training data (e.g., from tf.data.Dataset or numpy array)
    y=dummy_y_train,  # Your training labels
    epochs=3,         # Number of training epochs (reduced for a quick demo)
    batch_size=batch_size, # Batch size for training
    validation_data=(dummy_x_val, dummy_y_val) # Your validation data and labels
)


# Get training history from the history object
hist = history.history

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist['loss'], label='Training Loss')
plt.plot(hist['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(hist['accuracy'], label='Training Accuracy')
plt.plot(hist['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nEvaluating the model on the dummy validation set...")

loss, accuracy = model.evaluate(dummy_x_val, dummy_y_val)

print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

print("\nMaking predictions on new dummy data...")

# Generate a small batch of dummy data for inference
dummy_inference_data = np.random.rand(5, *input_shape).astype(np.float32)

# Make predictions
predictions = model.predict(dummy_inference_data)

print("Predicted probabilities for the first 5 samples:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {pred}")

# To get the predicted class labels, you can use np.argmax
predicted_labels = np.argmax(predictions, axis=-1)
print("\nPredicted class labels for the first 5 samples:")
print(predicted_labels)