from tensorflow.keras.layers import Lambda, Dense, Reshape, Input,Conv2D
from tensorflow.keras.models import Model
from layers import *

def build_capsule_net(input_shape, num_primary_capsules, dim_capsule, num_classes, dim_digit_capsule, routing_iterations):
    inputs = Input(shape=input_shape, name='input_image')

    # Standard Convolution Layer
    conv1 = Conv2D(
        filters=256,
        kernel_size=9,
        strides=1,
        activation='relu',
        name='conv1'
    )(inputs)

    # Primary Capsule Convolution Layer
    primary_caps_conv = Conv2D(
        filters=num_primary_capsules * dim_capsule, # e.g., 32 * 8 = 256 filters
        kernel_size=9,
        strides=2,
        padding='valid',
        activation='relu', 
        name='primary_caps_conv'
    )(conv1)

    # FIXED: Replaced hardcoded (6 * 6 * num_primary_capsules) with -1 to handle any input shape dynamically
    primary_capsules_reshaped = Reshape(
        target_shape=(-1, dim_capsule),
        name='primary_capsules_reshape'
    )(primary_caps_conv)

    # Apply Squash Activation
    primary_capsules_output = Lambda(squash, name='primary_capsules_squash')(primary_capsules_reshaped)

    # Digit Capsule Layer with Routing
    digit_capsules_output = DigitCaps(
        num_capsule=num_classes,
        dim_capsule=dim_digit_capsule,
        routing_iterations=routing_iterations,
        name='digit_capsules'
    )(primary_capsules_output)

    # Target Input for Reconstruction Masking
    y_true = Input(shape=(num_classes,), name='y_true') 

    # Masking Layer to isolate the target capsule's features
    # Lambda handles unpacking the list inside Masking_layer
    decoder_input = Lambda(Masking_layer, name='decoder_input')([digit_capsules_output, y_true])

    # Decoder Network for Reconstruction
    image_pixels = input_shape * input_shape * input_shape

    decoder = Dense(512, activation='relu', name='decoder_dense_1')(decoder_input)
    decoder = Dense(1024, activation='relu', name='decoder_dense_2')(decoder)
    reconstruction_output = Dense(image_pixels, activation='sigmoid', name='decoder_output')(decoder)

    # Reshape the output back to match the original input image dimensions
    reconstruction_output = Reshape(target_shape=input_shape, name='reconstruction_reshape')(reconstruction_output)

    # Build the Model
    model = Model(inputs=[inputs, y_true], outputs=[digit_capsules_output, reconstruction_output])
    return model