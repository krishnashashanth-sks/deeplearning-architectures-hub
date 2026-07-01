import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import *
import kagglehub
import torch
import os
import tensorflow as tf

# Hyperparameters used in training.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

vggish_keras_model = build_vggish_keras_model()

path = kagglehub.dataset_download("vbs2004/birdclef-audio-dataset")


# Construct the full path to the .pt file
birdclef_pt_path = os.path.join(path, 'birdclef_preprocessed.pt')

print(f"Attempting to load PyTorch file from: {birdclef_pt_path}")

try:
    # Load the PyTorch tensor
    # Use map_location to load it to CPU if no GPU is available or preferred
    loaded_data_torch = torch.load(birdclef_pt_path, map_location=torch.device('cpu'))

    # Inspect the loaded data
    print(f"Successfully loaded data from {birdclef_pt_path}")
    print(f"Type of loaded data: {type(loaded_data_torch)}")

    # Assuming the .pt file contains preprocessed features suitable for VGGish
    # If it's a tuple or dict, further inspection will be needed.
    if isinstance(loaded_data_torch, torch.Tensor):
        print(f"Shape of loaded PyTorch tensor: {loaded_data_torch.shape}")
        # Convert to NumPy array, then to TensorFlow tensor
        birdclef_features_np = loaded_data_torch.numpy()
        birdclef_features_tf = tf.convert_to_tensor(birdclef_features_np, dtype=tf.float32)
        print(f"Converted features to TensorFlow tensor with shape: {birdclef_features_tf.shape}")
        # Placeholder for labels, if any
        birdclef_labels_tf = None # Assuming .pt only contains features for now

    elif isinstance(loaded_data_torch, (tuple, list)) and len(loaded_data_torch) == 2:
        # Assuming it's (features, labels)
        features_torch, labels_torch = loaded_data_torch
        print(f"Loaded tuple/list: Features shape {features_torch.shape}, Labels shape {labels_torch.shape}")
        birdclef_features_tf = tf.convert_to_tensor(features_torch.numpy(), dtype=tf.float32)
        birdclef_labels_tf = tf.convert_to_tensor(labels_torch.numpy(), dtype=tf.int32)
        print(f"Converted features to TensorFlow tensor with shape: {birdclef_features_tf.shape}")
        print(f"Converted labels to TensorFlow tensor with shape: {birdclef_labels_tf.shape}")

    elif isinstance(loaded_data_torch, dict) and 'signals' in loaded_data_torch and 'labels' in loaded_data_torch:
        # Extract features and labels from the dictionary
        features_torch = loaded_data_torch['signals']
        labels_torch = loaded_data_torch['labels']
        print(f"Loaded dictionary: Features shape {features_torch.shape}, Labels shape {labels_torch.shape}")
        birdclef_features_tf = tf.convert_to_tensor(features_torch.numpy(), dtype=tf.float32)
        birdclef_labels_tf = tf.convert_to_tensor(labels_torch.numpy(), dtype=tf.int32)
        print(f"Converted features to TensorFlow tensor with shape: {birdclef_features_tf.shape}")
        print(f"Converted labels to TensorFlow tensor with shape: {birdclef_labels_tf.shape}")

    else:
        print("Loaded data is not a Tensor, a (features, labels) tuple/list, or a dict with 'signals' and 'labels'. Please inspect `loaded_data_torch` manually.")
        birdclef_features_tf = None
        birdclef_labels_tf = None

except Exception as e:
    print(f"Error loading or converting PyTorch file: {e}")
    birdclef_features_tf = None
    birdclef_labels_tf = None

# Assuming birdclef_features_tf now holds the data ready for further processing
if birdclef_features_tf is not None:
    print("BirdCLEF data successfully prepared for TensorFlow pipeline.")
else:
    print("Failed to prepare BirdCLEF data.")

num_classes = int(tf.reduce_max(birdclef_labels_tf) + 1)


classification_model=build_vggish_classfication_model(num_classes=num_classes)
# Reshape features to (num_samples, 96, 64, 1)
birdclef_features_tf_reshaped = tf.squeeze(birdclef_features_tf, axis=1)
birdclef_features_tf_reshaped = tf.expand_dims(birdclef_features_tf_reshaped, axis=-1)

print(f"\nReshaped features shape for training: {birdclef_features_tf_reshaped.shape}")

# Create a tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((birdclef_features_tf_reshaped, birdclef_labels_tf))

# Determine dataset size for splitting
dataset_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Shuffle and split into training and validation sets
dataset = dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=True)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Batch and prefetch the datasets for performance
batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Compile the classification model
classification_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("\nStarting training...")

# Train the model
epochs = 10 # You can adjust the number of epochs
history = classification_model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset
)

print("\nTraining complete.")


print("Performing inference on the validation set...")

# Get predictions on the validation dataset
y_pred_probs = classification_model.predict(val_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

# Extract true labels from the validation dataset
y_true = []
# Iterate over batches directly (val_dataset is already batched)
for _, labels_batch in val_dataset:
    y_true.append(labels_batch.numpy())
y_true = np.concatenate(y_true)

print(f"Number of validation samples: {len(y_true)}")
print(f"Number of predictions: {len(y_pred)}")

print("\n--- Visualization ---")

# 1. Confusion Matrix
print("Generating Confusion Matrix...")
# Due to the large number of classes (206), a fully annotated heatmap might be unreadable.
# We will generate it without annotations, or consider a smaller subset if needed.
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Validation Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 2. Classification Report
print("\nClassification Report (Validation Set):")
# Precision, recall, f1-score for each class
print(classification_report(y_true, y_pred, digits=4))

# 3. Visualize a few example predictions
print("\nVisualizing a few example predictions:")
# Take a batch from the validation dataset
for features, labels in val_dataset.take(1):
    sample_features = features.numpy()
    sample_labels = labels.numpy()

    # Predict for this batch
    sample_pred_probs = classification_model.predict(features)
    sample_pred_labels = np.argmax(sample_pred_probs, axis=1)

    print("\n--- Sample Predictions ---")
    for i in range(min(5, len(sample_features))): # Show up to 5 examples
        print(f"Sample {i+1}:")
        print(f"  True Label: {sample_labels[i]}")
        print(f"  Predicted Label: {sample_pred_labels[i]}")
        # Display top 3 predicted probabilities and their corresponding class indices
        top_3_indices = np.argsort(sample_pred_probs[i])[-3:][::-1]
        top_3_values = np.sort(sample_pred_probs[i])[-3:][::-1]
        print(f"  Top 3 Predicted (Class Index: Probability): {list(zip(top_3_indices, top_3_values))}")

        # Visualize the spectrogram
        plt.figure(figsize=(6, 4))
        # The spectrogram is (96, 64, 1), we want to plot (96, 64) and transpose for better visualization
        plt.imshow(sample_features[i, :, :, 0].T, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"True: {sample_labels[i]}, Pred: {sample_pred_labels[i]}")
        plt.xlabel('Time Frames')
        plt.ylabel('Mel Bins')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()