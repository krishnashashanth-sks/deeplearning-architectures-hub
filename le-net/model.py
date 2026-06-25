import tensorflow as tf
from tensorflow.keras import layers, models

def build_lenet5_model(input_shape=(32, 32, 1), num_classes=10):
    """
    Builds an advanced LeNet-5 convolutional neural network model.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
                             Default for LeNet is usually (32, 32, 1) for grayscale.
        num_classes (int): The number of output classes for classification.

    Returns:
        tf.keras.Model: The compiled LeNet-5 model.
    """
    model = models.Sequential([
        # C1 Convolutional Layer
        layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape,
                      padding='valid', name='C1_Conv2D'),
        # S2 Pooling Layer
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='S2_AvgPool'),

        # C3 Convolutional Layer
        layers.Conv2D(16, kernel_size=(5, 5), activation='relu', padding='valid', name='C3_Conv2D'),
        # S4 Pooling Layer
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='S4_AvgPool'),

        # Flatten Layer
        layers.Flatten(name='Flatten'),

        # F5 Fully Connected Layer
        layers.Dense(120, activation='relu', name='F5_Dense'),

        # F6 Fully Connected Layer
        layers.Dense(84, activation='relu', name='F6_Dense'),

        # Output Layer
        layers.Dense(num_classes, activation='softmax', name='Output_Dense')
    ])

    return model
main