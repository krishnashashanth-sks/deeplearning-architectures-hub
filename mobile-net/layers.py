import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, ZeroPadding2D, GlobalAveragePooling2D, Dense

def _inverted_res_block(inputs, expansion, stride, filters, block_id, alpha=1.0):
  in_channels = inputs.shape[-1] # Corrected to get last dimension for channels
  pointwise_conv_filters = int(filters * alpha)

  # Expansion convolution
  if expansion != 1:
    x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, name=f"expanded_conv_{block_id}_expand")(inputs)
    x = BatchNormalization(name=f'expanded_conv_{block_id}_expand_BN')(x) # Corrected name
    x = ReLU(6., name=f'expanded_conv_{block_id}_expand_ReLU')(x) # Corrected name
  else:
    
    x = inputs

  # Depthwise convolution
  x = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same' if stride == 1 else 'valid', use_bias=False, name=f'expanded_conv_{block_id}_depthwise')(x) # Fixed typo, kernel_size, and syntax
  x = BatchNormalization(name=f'expanded_conv_{block_id}_depthwise_BN')(x)
  x = ReLU(6., name=f'expanded_conv_{block_id}_depthwise_ReLU')(x) # Fixed depth_id to block_id

  # Projection convolution
  x = Conv2D(pointwise_conv_filters, kernel_size=1, padding='same', use_bias=False, name=f'expanded_conv_{block_id}_project')(x)
  x = BatchNormalization(name=f'expanded_conv_{block_id}_project_BN')(x)

  # Residual connection
  if stride == 1 and in_channels == pointwise_conv_filters: # Corrected condition from != to ==
    x = tf.keras.layers.Add(name=f'expanded_conv_{block_id}_add')([inputs, x]) # Corrected name

  return x
