from tensorflow.keras.datasets import mnist
import numpy as np

# 1. Load the training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 3. Reshape images to include a channel dimension (for grayscale images, channel is 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 4. Convert labels to float32
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
