from tensorflow.keras import layers

def inception_block(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    # 1x1 convolution path
    path_1x1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

    # 3x3 convolution path (1x1 reduction + 3x3 convolution)
    path_3x3 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    path_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(path_3x3)

    # 5x5 convolution path (1x1 reduction + 5x5 convolution)
    path_5x5 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    path_5x5 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(path_5x5)

    # Max pooling path (3x3 max pool + 1x1 projection)
    path_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path_pool = layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(path_pool)

    # Concatenate all parallel paths
    return layers.concatenate([path_1x1, path_3x3, path_5x5, path_pool], axis=-1)