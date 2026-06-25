from model import build_advanced_dannet
import numpy as np
import matplotlib.pyplot as plt
from inference import predict_with_dannet
from tensorflow import keras

input_shape = (32, 32, 3) # e.g., CIFAR-10 image size
num_classes = 10         # e.g., CIFAR-10 number of classes

model = build_advanced_dannet(input_shape, num_classes)
model.summary()

image_size = (32, 32)
batch_size = 32

# Generate dummy data
x_train = np.random.rand(1000, *image_size, 3).astype('float32') # 1000 samples, 32x32 RGB
y_train = np.random.randint(0, num_classes, size=(1000,)).astype('int32')

x_test = np.random.rand(100, *image_size, 3).astype('float32')
y_test = np.random.randint(0, num_classes, size=(100,)).astype('int32')

print(f"X_train shape: {x_train.shape}, Y_train shape: {y_train.shape}")
print(f"X_test shape: {x_test.shape}, Y_test shape: {y_test.shape}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

epochs = 5 # For demonstration, a small number of epochs
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1, # Use a small portion of training data for validation
    verbose=1
)

# Plotting training history
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

model = build_advanced_dannet(input_shape, num_classes)

# The model must also be compiled with the same optimizer and loss function used during training
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Note: If evaluating a truly 'trained' model after a kernel restart,
# you would need to load its weights here (e.g., model.load_weights('path/to/weights.h5')).
# Since this example uses dummy data and doesn't save weights,
# this evaluation will be on an untrained, freshly built model if run out of sequence.

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Let's create some dummy new data for inference (e.g., 5 new images)
num_inference_samples = 5
new_image_size = (32, 32, 3) # Should match the input_shape used for the model
new_data = np.random.rand(num_inference_samples, *new_image_size).astype('float32')

print(f"New data shape for inference: {new_data.shape}")

# Perform inference
inference_predictions = predict_with_dannet(model, new_data)

print("\nInference Predictions (first 2 samples):")
for i in range(min(2, num_inference_samples)): # Print predictions for the first 2 samples
  predicted_class = np.argmax(inference_predictions[i])
  confidence = inference_predictions[i][predicted_class]
  print(f"  Sample {i+1}: Predicted Class = {predicted_class}, Confidence = {confidence:.4f}")