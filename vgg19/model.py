from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def build_model(input_shape,num_classes):
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name="block1_conv1")(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name="block1_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name="block2_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name="block3_conv3")(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name="block3_conv4")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4 (Note: for 32x32 input, after 3 pooling layers, feature map is 4x4. 
    # One more pooling layer might make it 2x2. Two more pooling layers (Block 4 & 5) 
    # would make it 1x1 and then fractional, which is not ideal. 
    # Let's adapt by removing Block 5 for 32x32 input and adjusting Block 4 pooling.)
    # We will use only 3 pooling layers to ensure a reasonable feature map size for FC layers.
    # After block 3 pool: (32/2/2/2) = 4x4
    # So Block 4 pool will make it 2x2. We will skip Block 5 to avoid 1x1 before flattening.

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name="block4_conv1")(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name="block4_conv2")(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name="block4_conv3")(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name="block4_conv4")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x) # Feature map becomes 2x2

    # Removed Block 5 to avoid overly small feature maps for 32x32 input

    # Classification block
    x = Flatten(name='flatten')(x)
    # Adjust Dense layer sizes if 4096 is too large for MNIST
    x = Dense(1024, activation='relu', name='fc1')(x) # Reduced from 4096
    x = Dense(1024, activation='relu', name='fc2')(x) # Reduced from 4096
    output = Dense(num_classes, activation='softmax', name='predictions')(x)

    return Model(inputs=img_input, outputs=output, name='vgg19_mnist')
