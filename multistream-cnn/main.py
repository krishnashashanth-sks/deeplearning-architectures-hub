import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import build_model
from inference import predict_multistream_cnn

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 2. Normalize the image data to [0, 1] and convert to float32
x_train_normalized = x_train.astype('float32') / 255.0
x_test_normalized = x_test.astype('float32') / 255.0

# 3. For Stream 1 input: Resize original CIFAR-10 images to (64, 64, 3)
x_train_stream1 = tf.image.resize(x_train_normalized, (64, 64))
x_test_stream1 = tf.image.resize(x_test_normalized, (64, 64))

# 4. For Stream 2 input: Convert to grayscale and resize to (32, 32, 1)
x_train_gray = tf.image.rgb_to_grayscale(x_train_normalized)
x_train_stream2 = tf.image.resize(x_train_gray, (32, 32))

x_test_gray = tf.image.rgb_to_grayscale(x_test_normalized)
x_test_stream2 = tf.image.resize(x_test_gray, (32, 32))

# 5. One-hot encode the target labels
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=10)

multistream_cnn_model=build_model(input_shape_stream1=(64,64,3),input_shape_stream2=(32,32,1))
multistream_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = multistream_cnn_model.fit(
    [x_train_stream1, x_train_stream2],
    y_train_encoded,
    epochs=10,
    batch_size=32,
    validation_data=([x_test_stream1, x_test_stream2], y_test_encoded)
)
# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

print("Training history visualized successfully.")
# Select a single sample from the test set for demonstration
sample_index = 0

# Prepare input for stream 1 (add batch dimension)
sample_stream1 = np.expand_dims(x_test_stream1[sample_index], axis=0)

# Prepare input for stream 2 (add batch dimension)
sample_stream2 = np.expand_dims(x_test_stream2[sample_index], axis=0)

# Get predictions using the defined function
predictions = predict_multistream_cnn(multistream_cnn_model,sample_stream1, sample_stream2)

print(f"Shape of predictions: {predictions.shape}")
print(f"Predictions for sample {sample_index}: {predictions[0]}")

# Optionally, print the true label for comparison
true_label_encoded = y_test_encoded[sample_index]
true_label_class = np.argmax(true_label_encoded)
print(f"True label for sample {sample_index}: {true_label_class}")
print("Prediction function tested successfully with a sample from the test dataset.")