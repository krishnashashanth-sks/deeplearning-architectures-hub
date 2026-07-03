import tensorflow as tf
import numpy as np
from main import input_shape

def preprocess_image(images):
    # Expand dimensions to (batch, height, width, channels=1)
    images = np.expand_dims(images, axis=-1)
    # Resize images to 32x32
    images = tf.image.resize(images, (input_shape[0], input_shape[1]))
    # Convert grayscale to RGB (3 channels) by stacking the single channel
    images = tf.image.grayscale_to_rgb(images)
    # Normalize pixel values to [0, 1]
    images = tf.cast(images, tf.float32) / 255.0
    return images