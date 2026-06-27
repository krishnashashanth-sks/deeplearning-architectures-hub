import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from model import build_capsule_net
from tensorflow.keras.optimizers import Adam
from losses import *
from dataset import y_train,y_test,x_train,x_test

# Define constants for the digit capsule layer
num_classes = 10         # Number of digit classes (0-9 for MNIST)
dim_digit_capsule = 16   # Dimension of each digit capsule vector (as per Hinton's paper)
routing_iterations = 3   # Number of routing iterations
input_shape = (28, 28, 1)
num_primary_capsules = 32 # Number of primary capsules
dim_capsule = 8   

model=build_capsule_net(input_shape,num_primary_capsules,dim_capsule,num_classes,dim_digit_capsule,routing_iterations)

reconstruction_loss_weight = 0.0005 * 784 # image_pixels is 784

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=[margin_loss, reconstruction_loss],
    loss_weights=[1., reconstruction_loss_weight],
    metrics={'digit_capsules': 'accuracy'}
)


y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

epochs = 10 
batch_size = 128 

history = model.fit(
    [x_train, y_train_one_hot],
    [y_train_one_hot, x_train],
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([x_test, y_test_one_hot],
                       [y_test_one_hot, x_test]
    ),
)


# --- Model Inference and Usage ---

# Assuming x_test and y_test_one_hot are already loaded and preprocessed
# and 'model' is the trained Capsule Network model.

print("Performing inference on test data...")

# Predict on a batch of test data
# model.predict returns two outputs: digit_capsules_output and reconstruction_output
digit_caps_pred, reconstructed_images_pred = model.predict([x_test, np.zeros((x_test.shape[0], num_classes))])

# Interpretation of Classification Output (digit_caps_pred):
# The length of each capsule vector in the digit_capsules_output represents the probability
# that the corresponding digit is present. We can find the predicted class by taking the argmax
# of these lengths.

# Calculate the length of each digit capsule vector
# (batch_size, num_classes)
digit_caps_lengths = np.sqrt(np.sum(np.square(digit_caps_pred), axis=-1))

# Get the predicted class (index with the largest length)
predicted_classes = np.argmax(digit_caps_lengths, axis=1)

# Get the true classes from the one-hot encoded y_test
true_classes = np.argmax(y_test_one_hot, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nExample Predictions (First 5 test samples):")
for i in range(5):
    print(f"  Sample {i+1}: True Class: {true_classes[i]}, Predicted Class: {predicted_classes[i]}")

# You can also visualize some reconstructions if desired (requires matplotlib)
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 4))
# for i in range(5):
#     # Original image
#     ax = plt.subplot(2, 5, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
#     plt.title(f"True: {true_classes[i]}")
#     plt.axis('off')

#     # Reconstructed image
#     ax = plt.subplot(2, 5, i + 1 + 5)
#     plt.imshow(reconstructed_images_pred[i].reshape(28, 28), cmap='gray')
#     plt.title(f"Pred: {predicted_classes[i]}")
#     plt.axis('off')
# plt.suptitle("Original vs. Reconstructed Images (First 5)")
# plt.show()
