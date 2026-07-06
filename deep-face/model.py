from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.models import Model

def build_deepface_model(input_shape=(152, 152, 3), embedding_dim=128):
    img_input = Input(shape=input_shape)

    # Initial Convolutional and Pooling Layers
    x = Conv2D(64, (11, 11), activation='relu', name='conv1', strides=(4, 4))(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    x = Conv2D(160, (5, 5), activation='relu', name='conv2', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool2')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    # Approximation of Locally Connected Layers
    # The original DeepFace paper uses LocallyConnected2D, which is computationally expensive
    # These are approximated with Conv2D layers with padding='same' to maintain spatial dimensions
    # after the initial pooling reduces them significantly (e.g., to 1x1 or 2x2 for original input).
    # Given the input_shape of 152x152, after pool3, spatial dimensions are likely 2x2 or 1x1.
    # Let's adjust padding to 'valid' if the input size is already large enough for `valid` to make sense
    # or 'same' if we want to preserve size for these specific layers.
    # The previous trace suggests x has shape (None, 1, 1, 256) before these layers,
    # so `padding='same'` is crucial here.
    x = Conv2D(256, (3, 3), activation='relu', name='LConv4_approx', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, (1, 1), activation='relu', name='LConv5_approx', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, (1, 1), activation='relu', name='LConv6_approx', strides=(1, 1), padding='same')(x)

    # Flatten and Embedding Layer
    x = Flatten(name='Flatten')(x)
    embedding = Dense(embedding_dim, activation='relu', name='Embedding')(x)

    model = Model(inputs=img_input, outputs=embedding, name='DeepFace_Model_Approx')
    return model