from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='best_deepfake_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)
# Define a simple convolutional model for deepfake detection.
# This model assumes input images are processed face images, e.g., 224x224x3 (RGB).
input_shape = (224, 224, 3) # Example input shape for a single processed face image

deepfake_detector_model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Binary classification
])

deepfake_detector_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create simulated data generators using tf.data.Dataset for compatibility with model.fit.
# These simulate processed video frames or features for training and validation.

num_samples_train = 100
num_samples_val = 20
batch_size = 8

# Generate simulated image-like data
simulated_train_data = np.random.rand(num_samples_train, *input_shape).astype(np.float32)
simulated_train_labels = np.random.randint(0, 2, size=num_samples_train).astype(np.float32)
simulated_val_data = np.random.rand(num_samples_val, *input_shape).astype(np.float32)
simulated_val_labels = np.random.randint(0, 2, size=num_samples_val).astype(np.float32)

# Create data generators
train_generator = tf.data.Dataset.from_tensor_slices((simulated_train_data, simulated_train_labels)).batch(batch_size).repeat()
val_generator = tf.data.Dataset.from_tensor_slices((simulated_val_data, simulated_val_labels)).batch(batch_size).repeat()

# Calculate steps per epoch for the generators
steps_per_epoch_train = num_samples_train // batch_size
steps_per_epoch_val = num_samples_val // batch_size

history = deepfake_detector_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=50,
    validation_data=val_generator,
    validation_steps=steps_per_epoch_val,
    callbacks=[early_stopping, model_checkpoint]
)

print("Model training initiated with simulated data generators.")
print(f"Training history keys: {history.history.keys()}")
# Make predictions on the validation data
# Since val_generator is a tf.data.Dataset and repeats, we need to specify steps
val_predictions_raw = deepfake_detector_model.predict(val_generator, steps=steps_per_epoch_val)

# Convert probabilities to binary predictions (0 or 1)
val_predictions = (val_predictions_raw > 0.5).astype(int)

# Get true labels from the validation generator
# We need to iterate through the generator to get the labels
# Resetting generator or creating a new one to get labels in order
val_labels = []
for _ in range(steps_per_epoch_val):
    _, labels_batch = next(iter(val_generator))
    val_labels.extend(labels_batch.numpy())
val_labels = np.array(val_labels)

# Calculate metrics
accuracy = accuracy_score(val_labels, val_predictions)
precision = precision_score(val_labels, val_predictions)
recall = recall_score(val_labels, val_predictions)
f1 = f1_score(val_labels, val_predictions)

print(f"\nModel Evaluation on Validation Data:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")