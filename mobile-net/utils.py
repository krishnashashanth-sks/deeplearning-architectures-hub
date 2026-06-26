import tensorflow as tf

def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
    return image, label