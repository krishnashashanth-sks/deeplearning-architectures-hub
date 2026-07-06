from layers import DeepLabV3PlusDecoder,deeplabv3_plus_encoder
import tensorflow as tf

# --- 7. DeepLabV3+ Model Function ---
def DeepLabV3Plus(input_shape=(256, 256, 3), num_classes=21, output_stride=16, aspp_filters=256, aspp_dilation_rates=[6, 12, 18]):
    inputs = tf.keras.Input(shape=input_shape)

    high_level_features, low_level_features = deeplabv3_plus_encoder(
        input_shape=input_shape,
        output_stride=output_stride,
        aspp_filters=aspp_filters,
        aspp_dilation_rates=aspp_dilation_rates
    )(inputs)

    decoder = DeepLabV3PlusDecoder(num_classes=num_classes)

    decoder_output = decoder([high_level_features, low_level_features])

    original_height, original_width = input_shape[0], input_shape[1]

    upsampling_layer = BilinearUpsampling(
        target_size=(original_height, original_width),
        name='final_upsample'
    )
    output_segmentation_map = upsampling_layer(decoder_output)

    model = tf.keras.Model(inputs=inputs, outputs=output_segmentation_map, name='DeepLabV3Plus')

    return model