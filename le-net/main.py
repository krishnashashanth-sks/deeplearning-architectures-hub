from dataset import x_train,y_train

# Define input shape (e.g., for MNIST 28x28 images, you might want to pad to 32x32)
input_shape = (32, 32, 1) # Assuming grayscale images padded to 32x32
num_classes = 10 # For MNIST digits 0-9

# Build the model
lenet_model = build_lenet5_model(input_shape=input_shape, num_classes=num_classes)

# Display the model summary
print("\nLeNet-5 Model Summary:")
lenet_model.summary()

# Compile the model (example configuration)
lenet_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = lenet_model.fit(
    x_train, y_train,
    epochs=10,  # You can adjust the number of epochs
    batch_size=128, # You can adjust the batch size
    validation_split=0.1, # Use 10% of training data for validation
    verbose=1
)

loss, accuracy = lenet_model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

import numpy as np
import matplotlib.pyplot as plt

print("\n--- Performing Model Inference ---")

# Make predictions on the test set
predictions = lenet_model.predict(x_test)

# Convert predictions from probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Display some example predictions
num_examples = 5
plt.figure(figsize=(10, 5))

for i in range(num_examples):
    # Select a random image from the test set
    idx = np.random.randint(0, len(x_test))

    plt.subplot(1, num_examples, i + 1)
    # Convert EagerTensor to NumPy array before reshaping
    plt.imshow(x_test[idx].numpy().reshape(32, 32), cmap='gray')
    plt.title(f"True: {y_test[idx]}\nPred: {predicted_labels[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.show()