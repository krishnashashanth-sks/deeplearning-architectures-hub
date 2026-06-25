import tensorflow as tf

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess the data

# Reshape to (num_samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')

# Pad images to 32x32 as required by LeNet-5 input_shape=(32, 32, 1)
x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

# Normalize pixel values to [0, 1]
x_train /= 255.0
x_test /= 255.0