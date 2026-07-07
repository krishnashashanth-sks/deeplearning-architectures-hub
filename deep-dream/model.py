from tensorflow import keras
from tensorflow.keras import layers
from layers import inception_block

def build_custom_inception_model(input_shape=(128, 128, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # First Inception Block
    x = inception_block(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)

    # Second Inception Block
    x = inception_block(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Third Inception Block
    x = inception_block(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model_custom_inception = keras.Model(inputs=inputs, outputs=outputs)
    return model_custom_inception

