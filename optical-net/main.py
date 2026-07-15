import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import build_model
import tensorflow as tf

#1. Model Initialization
model=build_model()
print("Advanced Custom ONN Architecture Summary:")
model.summary()

# 2. Instantiate EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 3. Instantiate ModelCheckpoint
model_checkpoint = ModelCheckpoint(
    filepath='best_advanced_onn_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 4. Instantiate ReduceLROnPlateau
reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

# 5. Create a list of callbacks
callbacks = [early_stopping, model_checkpoint, reduce_lr_on_plateau]

#6.Model compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7.Load CIFAR-10 dataset
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
X_train_raw = X_train_raw.astype('float32') / 255.0
X_test_raw = X_test_raw.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test_raw, num_classes=10)

# Define target image size for the model
image_height = 64
image_width = 64
image_channels = 3 # CIFAR-10 images are 32x32x3
num_classes = 10
batch_size = 32

# Function to resize images
def resize_image(image):
    return tf.image.resize(image, (image_height, image_width))

# Apply resizing to the datasets
X_train = np.array([resize_image(img) for img in X_train_raw])
X_test = np.array([resize_image(img) for img in X_test_raw])

# Split X_test into validation and test sets
val_split_ratio = 0.5
num_test_samples = X_test.shape[0]
num_val_samples = int(num_test_samples * val_split_ratio)

X_val = X_test[:num_val_samples]
y_val = y_test[:num_val_samples]

X_test = X_test[num_val_samples:]
y_test = y_test[num_val_samples:]

print("CIFAR-10 data loaded and preprocessed.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# 8.Dataset Prepation from raw
# 1. Convert to tf.data.Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

print("Datasets converted to tf.data.Dataset objects.")

# 2. Shuffle and Batch datasets
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

print(f"Datasets shuffled (train) and batched with batch_size={batch_size}.")

# 3. Optimize datasets with prefetch
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# 9.Training
history = model.fit(
    train_dataset,
    epochs=1, # You can adjust the number of epochs
    validation_data=val_dataset,
    callbacks=callbacks
)

# 10.Inference
y_pred_probs = model.predict(test_dataset)
# Map CIFAR-10 numerical labels to class names for better readability
cifar10_class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 2. Get the actual class labels from y_test
y_true_labels = np.argmax(y_test, axis=1)

# 3. Get the predicted class labels from y_pred_probs
y_predicted_labels = np.argmax(y_pred_probs, axis=1)

# 4. Select a small number of random indices from the test set
num_images_to_display = 10
random_indices = np.random.choice(len(X_test), num_images_to_display, replace=False)

print(f"Displaying {num_images_to_display} random images from the test set with predictions:")

# 5. Iterate through the selected indices and display images
plt.figure(figsize=(15, 8))
for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i + 1) # Arrange plots in 2 rows, 5 columns
    plt.imshow(X_test[idx])

    true_label_name = cifar10_class_names[y_true_labels[idx]]
    predicted_label_name = cifar10_class_names[y_predicted_labels[idx]]

    color = "green" if true_label_name == predicted_label_name else "red"
    plt.title(f"True: {true_label_name}\nPred: {predicted_label_name}", color=color)
    plt.axis('off')

# 6. Use plt.tight_layout() to prevent labels from overlapping and plt.show() to display the plots.
plt.tight_layout()
plt.show()