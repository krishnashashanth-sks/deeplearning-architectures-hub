from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

class AtrousConvolution(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', activation=None, dilation_rate=1, **kwargs):
        super(AtrousConvolution, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            dilation_rate=self.dilation_rate
        )

    def build(self, input_shape):
        super(AtrousConvolution, self).build(input_shape)

    def call(self, inputs):
        return self.conv(inputs)

    def get_config(self):
        config = super(AtrousConvolution, self).get_config()
        config.update(
            {'filters': self.filters,
             'kernel_size': self.kernel_size,
             'strides': self.strides,
             'padding': self.padding,
             'activation': self.activation,
             'dilation_rate': self.dilation_rate
             })
        return config


# --- 2. ASPP Layer ---
class ASPP(layers.Layer):
    def __init__(self, filters, dilation_rates, **kwargs):
        super(ASPP, self).__init__(**kwargs)
        self.filters = filters
        self.dilation_rates = dilation_rates
        self.conv1x1_branch = layers.Conv2D(
            filters=self.filters,
            kernel_size=1, padding='same',
            activation='relu'
        )
        self.atrous_conv_branches = []
        for rate in self.dilation_rates:
            self.atrous_conv_branches.append(AtrousConvolution(
                filters=self.filters,
                kernel_size=3,
                dilation_rate=rate,
                padding='same',
                activation='relu'
            ))
        self.global_average_pooling_branch = layers.GlobalAveragePooling2D()
        self.gap_conv1x1_branch = layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            padding='same',
            activation='relu'
        )
        self.output_conv1x1 = layers.Conv2D(
            filters=self.filters,
            kernel_size=1, padding='same',
            activation='relu'
        )

    def build(self, input_shape):
        super(ASPP, self).build(input_shape)

    def call(self, inputs):
        output_1x1 = self.conv1x1_branch(inputs)
        outputs_atrous = [conv_layer(inputs) for conv_layer in self.atrous_conv_branches]
        pooled_output = self.global_average_pooling_branch(inputs)
        pooled_output = tf.expand_dims(pooled_output, 1)
        pooled_output = tf.expand_dims(pooled_output, 1)
        pooled_output = self.gap_conv1x1_branch(pooled_output)
        input_shape = tf.shape(inputs)
        h, w = input_shape[1], input_shape[2]
        pooled_output = tf.image.resize(pooled_output, (h, w), method='bilinear')
        concatenated_outputs = tf.concat([output_1x1] + outputs_atrous + [pooled_output], axis=-1)
        return self.output_conv1x1(concatenated_outputs)

    def get_config(self):
        config = super(ASPP, self).get_config()
        config.update({
            'filters': self.filters,
            'dilation_rates': self.dilation_rates
        })
        return config


# --- 3. Modified ResNet50 Backbone (resnet_atrous_backbone) ---
def resnet_atrous_backbone(input_shape=(256, 256, 3), output_stride=16, weights='imagenet'):
    def _resnet_conv_block(input_tensor, filters, kernel_size, stage, block, strides=(2, 2), dilation_rate=(1, 1), use_bias=False):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        filters1, filters2, filters3 = filters

        x = layers.Conv2D(filters1, (1, 1), strides=strides,
                          name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
        x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same',
                          dilation_rate=dilation_rate, name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
        x = layers.BatchNormalization(name=bn_name_base + '2c')(x)

        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                                 name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
        shortcut = layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

    def _resnet_identity_block(input_tensor, filters, kernel_size, stage, block, dilation_rate=(1, 1), use_bias=False):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        filters1, filters2, filters3 = filters

        x = layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
        x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same',
                          dilation_rate=dilation_rate, name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
        x = layers.BatchNormalization(name=bn_name_base + '2c')(x)

        x = layers.Add()([x, input_tensor])
        x = layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

    img_input = tf.keras.Input(shape=input_shape)

    x = img_input
    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', use_bias=False, name='conv1_conv')(x)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = _resnet_conv_block(x, [64, 64, 256], (3, 3), stage=2, block='block1', strides=(1, 1), use_bias=False)
    x = _resnet_identity_block(x, [64, 64, 256], (3, 3), stage=2, block='block2', use_bias=False)
    x = _resnet_identity_block(x, [64, 64, 256], (3, 3), stage=2, block='block3', use_bias=False)
    low_level_features = x

    conv3_strides = (2, 2) if output_stride <= 16 else (1, 1)
    conv3_dilation_rate = (1, 1) if output_stride <= 16 else (2, 2)

    x = _resnet_conv_block(x, [128, 128, 512], (3, 3), stage=3, block='block1', strides=conv3_strides, dilation_rate=(1, 1), use_bias=False)
    x = _resnet_identity_block(x, [128, 128, 512], (3, 3), stage=3, block='block2', dilation_rate=conv3_dilation_rate, use_bias=False)
    x = _resnet_identity_block(x, [128, 128, 512], (3, 3), stage=3, block='block3', dilation_rate=conv3_dilation_rate, use_bias=False)
    x = _resnet_identity_block(x, [128, 128, 512], (3, 3), stage=3, block='block4', dilation_rate=conv3_dilation_rate, use_bias=False)

    conv4_strides = (2, 2) if output_stride <= 8 else (1, 1)
    conv4_dilation_rate = (1, 1) if output_stride <= 8 else (2, 2)

    x = _resnet_conv_block(x, [256, 256, 1024], (3, 3), stage=4, block='block1', strides=conv4_strides, dilation_rate=(1, 1), use_bias=False)
    for i in range(2, 7):
        x = _resnet_identity_block(x, [256, 256, 1024], (3, 3), stage=4, block=f'block{i}', dilation_rate=conv4_dilation_rate, use_bias=False)

    conv5_dilation_rate = (1, 1)
    if output_stride == 16:
        conv5_dilation_rate = (2, 2)
    elif output_stride == 8:
        conv5_dilation_rate = (4, 4)

    x = _resnet_conv_block(x, [512, 512, 2048], (3, 3), stage=5, block='block1', strides=(1, 1), dilation_rate=conv5_dilation_rate, use_bias=False)
    x = _resnet_identity_block(x, [512, 512, 2048], (3, 3), stage=5, block='block2', dilation_rate=conv5_dilation_rate, use_bias=False)
    x = _resnet_identity_block(x, [512, 512, 2048], (3, 3), stage=5, block='block3', dilation_rate=conv5_dilation_rate, use_bias=False)

    model = tf.keras.Model(img_input, [x, low_level_features], name='resnet50_atrous_backbone')

    if weights == 'imagenet':
        temp_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        for layer in model.layers:
            try:
                if layer.name == 'tf.keras.Input':
                    continue
                temp_layer = temp_model.get_layer(name=layer.name)
                layer.set_weights(temp_layer.get_weights())
            except ValueError:
                pass
    return model


# --- 4. BilinearUpsampling Layer ---
class BilinearUpsampling(layers.Layer):
    def __init__(self, target_size, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(
            inputs,
            self.target_size,
            method='bilinear'
        )

    def get_config(self):
        config = super(BilinearUpsampling, self).get_config()
        config.update(
            {'target_size': self.target_size}
        )
        return config


# --- 5. DeepLabV3PlusDecoder Layer ---
class DeepLabV3PlusDecoder(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(DeepLabV3PlusDecoder, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.low_level_conv = layers.Conv2D(filters=48, kernel_size=1, padding='same', activation='relu', name='decoder_low_level_conv')
        self.low_level_bn = layers.BatchNormalization(name='decoder_low_level_bn')
        self.decoder_conv1 = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name='decoder_conv1')
        self.decoder_bn1 = layers.BatchNormalization(name='decoder_bn1')
        self.decoder_conv2 = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name='decoder_conv2')
        self.decoder_bn2 = layers.BatchNormalization(name='decoder_bn2')
        self.output_conv = layers.Conv2D(filters=self.num_classes, kernel_size=1, padding='same', name='decoder_output_conv')

    def build(self, input_shape):
        super(DeepLabV3PlusDecoder, self).build(input_shape)

    def call(self, inputs):
        high_level_features, low_level_features = inputs
        low_level_features = self.low_level_conv(low_level_features)
        low_level_features = self.low_level_bn(low_level_features)

        target_height = tf.shape(low_level_features)[1]
        target_width = tf.shape(low_level_features)[2]

        high_level_features_upsampled = tf.image.resize(
            high_level_features,
            (target_height, target_width),
            method='bilinear'
        )

        concatenated_features = tf.concat([high_level_features_upsampled, low_level_features], axis=-1)

        x = self.decoder_conv1(concatenated_features)
        x = self.decoder_bn1(x)
        x = self.decoder_conv2(x)
        x = self.decoder_bn2(x)

        decoder_output = self.output_conv(x)

        return decoder_output

    def get_config(self):
        config = super(DeepLabV3PlusDecoder, self).get_config()
        config.update({
            'num_classes': self.num_classes
        })
        return config


# --- 6. DeepLabV3+ Encoder Function ---
def deeplabv3_plus_encoder(input_shape=(256, 256, 3), output_stride=16, aspp_filters=256, aspp_dilation_rates=[6, 12, 18]):
    backbone_model = resnet_atrous_backbone(input_shape=input_shape, output_stride=output_stride, weights='imagenet')
    high_level_features, low_level_features = backbone_model.output

    aspp_layer = ASPP(filters=aspp_filters, dilation_rates=aspp_dilation_rates)
    aspp_output = aspp_layer(high_level_features)

    encoder_model = tf.keras.Model(inputs=backbone_model.input, outputs=[aspp_output, low_level_features], name='deeplabv3_plus_encoder')

    return encoder_model
