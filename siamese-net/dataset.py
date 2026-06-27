import tensorflow as tf
import numpy as np
from utils import create_triplets

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

IMG_HEIGHT, IMG_WIDTH = x_train.shape[1], x_train.shape[2]

# Normalize pixel values to [0, 1] and reshape to add channel dimension
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

num_train_triplets = 60000
num_test_triplets = 10000

train_triplets = create_triplets(x_train, y_train, num_train_triplets)

test_triplets = create_triplets(x_test, y_test, num_test_triplets)

batch_size = 32

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
train_triplets).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices(
test_triplets).batch(batch_size).prefetch(tf.data.AUTOTUNE)
