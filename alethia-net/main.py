from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import build_alethia_advanced_model
from inference import infer_on_image

input_shape = (32, 32, 3) # Example input shape for image data
num_classes = 10 # Example number of classes
alethia_model = build_alethia_advanced_model(input_shape, num_classes)
alethia_model.summary()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoded vectors
num_classes = 10 # CIFAR-10 has 10 classes
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, test_size=0.15, random_state=42)

BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

# Create training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_split, y_train_split))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Create validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val_split, y_val_split))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Create test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("tf.data.Dataset objects created successfully: train_dataset, val_dataset, test_dataset.")
print("Defining learning rate schedule and compiling the model...")

# 1. Define a learning rate schedule
initial_learning_rate = 0.001
decay_steps = 10000 # Number of steps after which to decay the learning rate
decay_rate = 0.9 # Rate at which the learning rate decays

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)

# 2. Instantiate an Adam optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 3. Compile the alethia_model
alethia_model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

print("Model compiled successfully with Adam optimizer, exponential decay learning rate, and categorical crossentropy loss.")
print("Setting up callbacks and starting model training...")

# Define callbacks
# EarlyStopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10, # Number of epochs with no improvement after which training will be stopped.
    restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity.
)

# ModelCheckpoint to save the best model based on validation accuracy
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_alethia_model.keras', # Using .keras extension for the new Keras 3 format
    monitor='val_accuracy',
    save_best_only=True, # Only save when val_accuracy improves
    verbose=1
)

# Train the model
EPOCHS = 10# You might want to adjust this based on your early stopping patience
history = alethia_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping, model_checkpoint]
)

# Create a figure with two subplots
plt.figure(figsize=(12, 5))

# Plot Training and Validation Loss
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

# Ensure plots do not overlap and display them
plt.tight_layout()
plt.show()

print("Plots generated successfully.")

# Define CIFAR-10 class names (if not already defined)
cifar10_class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

print("Demonstrating inference on an example image from the test set...")

# Pick an example image from the test set
example_index = 0 # You can change this index to try different images
example_image = x_test[example_index]
true_label_index = np.argmax(y_test[example_index])
true_label_name = cifar10_class_names[true_label_index]

# Perform inference
predicted_name, predicted_prob = infer_on_image(alethia_model, example_image, cifar10_class_names)

# Display the image and prediction
plt.imshow(example_image)
plt.title(f"True: {true_label_name}, Predicted: {predicted_name} (Prob: {predicted_prob:.2f})")
plt.axis('off')
plt.show()

print(f"Inference complete for example image {example_index}.")
