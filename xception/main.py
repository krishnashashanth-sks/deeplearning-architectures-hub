import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize as tf_resize
import tensorflow as tf
from model import build_xception

# Example usage:
model = build_xception(input_shape=(299, 299, 3), num_classes=1000) # Standard ImageNet input size and classes
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Xception model built successfully!")
model.summary()
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define IMG_SIZE for later use in the tf.data pipeline
IMG_SIZE = 299

# Convert labels to one-hot encoding
NUM_CLASSES = 10
y_train_one_hot = to_categorical(y_train, NUM_CLASSES)
y_test_one_hot = to_categorical(y_test, NUM_CLASSES)

print(f"Original X_train shape: {x_train.shape}")
print(f"Y_train one-hot shape: {y_train_one_hot.shape}")
print(f"Number of classes: {NUM_CLASSES}")
# Rebuild the Xception model with CIFAR-10 specific parameters
model_cifar10 = build_xception(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)

# Compile the model
model_cifar10.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Xception model for CIFAR-10 built successfully!")
model_cifar10.summary()

# Create tf.data.Dataset objects for training and validation
BATCH_SIZE = 16 # Define a batch size (reduced from 32 to 16 to mitigate RAM deficiency)

# Define a resizing function to apply within the tf.data pipeline
def preprocess_image(image, label):
    image = tf_resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
.batch(BATCH_SIZE)
.prefetch(tf.data.AUTOTUNE))

test_dataset = (tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
.batch(BATCH_SIZE)
.prefetch(tf.data.AUTOTUNE))

# Train the model
EPOCHS = 10 # You might want to increase this for better performance

history = model_cifar10.fit(
    train_dataset, # Use the tf.data.Dataset for training
    epochs=EPOCHS,
    validation_data=test_dataset # Use the tf.data.Dataset for validation
)

print("Model training complete!")
print("Performing inference on the test dataset...")
predictions = model_cifar10.predict(test_dataset)

# Display the shape of the predictions
print(f"Shape of predictions: {predictions.shape}")

# Display the first 5 predictions (probabilities for each class)
print("First 5 predictions (probabilities):")
print(predictions[:5])

# Convert probabilities to predicted class labels for the first 5 samples
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test_one_hot, axis=1)

print("\nFirst 5 predicted class labels:")
print(predicted_labels[:5])
print("First 5 true class labels:")
print(true_labels[:5])

import matplotlib.pyplot as plt

# Get a batch of images and labels from the test dataset for visualization
# We'll take one batch to display a few examples
for images, labels in test_dataset.take(1):
    sample_images = images.numpy()
    sample_true_labels = np.argmax(labels.numpy(), axis=1)
    sample_predictions = model_cifar10.predict(images)
    sample_predicted_labels = np.argmax(sample_predictions, axis=1)

    plt.figure(figsize=(12, 12))
    for i in range(min(sample_images.shape[0], 9)): # Display up to 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(sample_images[i])
        plt.title(f"True: {sample_true_labels[i]}, Pred: {sample_predicted_labels[i]}")
        plt.axis('off')
    plt.suptitle("Sample Test Images with True and Predicted Labels", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

loss, accuracy = model_cifar10.evaluate(test_dataset, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")