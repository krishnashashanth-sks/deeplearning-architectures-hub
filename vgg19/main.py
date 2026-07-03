import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess_image
from model import build_model

# --- 1. Load and Preprocess MNIST Dataset ---
print("Loading and preprocessing MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# VGG19 expects 3-channel input, so convert MNIST (grayscale) to RGB and resize.
# Original MNIST images are 28x28. We'll resize to 32x32 to better fit VGG-like architectures.
input_shape = (32, 32, 3)
num_classes = 10 # MNIST has 10 classes (digits 0-9)

x_train_processed = preprocess_image(x_train)
x_test_processed = preprocess_image(x_test)

# Convert labels to one-hot encoding
y_train_categorical = to_categorical(y_train, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)

vgg19_mnist_model=build_model(input_shape,num_classes)

# --- 3. Compile the Model ---
print("Compiling the model...")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
vgg19_mnist_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# --- 4. Train the Model ---
print("Training the model...")
epochs = 5 # Reduced epochs for quicker demonstration
batch_size = 64

history = vgg19_mnist_model.fit(
    x_train_processed, y_train_categorical,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1 # Use a small validation split
)

print("Model training completed.")

# --- 5. Evaluate the Model ---
print("Evaluating the model...")
loss, accuracy = vgg19_mnist_model.evaluate(x_test_processed, y_test_categorical)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Optional: Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("--- Model Inference and Usage ---")

# Get a random sample from the test set for inference
# We'll take one image from the preprocessed test set and its corresponding true label.

sample_index = np.random.randint(0, len(x_test_processed))
sample_image = x_test_processed[sample_index]
sample_true_label_one_hot = y_test_categorical[sample_index]
sample_true_label = np.argmax(sample_true_label_one_hot) # Convert one-hot to single digit

# The model expects a batch of images, so expand dimensions
sample_image_batch = np.expand_dims(sample_image, axis=0)

# Make a prediction
predictions = vgg19_mnist_model.predict(sample_image_batch)
predicted_probability = predictions[0] # Get probabilities for the single sample
predicted_class = np.argmax(predicted_probability) # Get the class with the highest probability

print(f"Sample Image Index: {sample_index}")
print(f"True Label: {sample_true_label}")
print(f"Predicted Label: {predicted_class}")
print(f"Prediction Probabilities: {predicted_probability}")

# Display the image and prediction
plt.figure(figsize=(4, 4))
plt.imshow(sample_image) # Display the 3-channel (RGB) image
plt.title(f"True: {sample_true_label}, Predicted: {predicted_class}")
plt.axis('off')
plt.show()

# Optional: Display the top 3 predictions with probabilities
print("Top 3 Predictions:")
top_3_indices = np.argsort(predicted_probability)[::-1][:3]
for i in top_3_indices:
    print(f"Class {i}: {predicted_probability[i]*100:.2f}%")