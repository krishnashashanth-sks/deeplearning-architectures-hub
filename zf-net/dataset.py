import tensorflow as tf

# 1. Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# 2. Preprocess data
# One-hot encode labels for categorical_crossentropy
num_classes_cifar100 = 100
y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes_cifar100)
y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes_cifar100)

# Create tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_ohe))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test_ohe))

# Function to resize and rescale images
IMG_HEIGHT = 224
IMG_WIDTH = 224
batch_size = 2# Reduced batch size to 16

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image, label

train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

