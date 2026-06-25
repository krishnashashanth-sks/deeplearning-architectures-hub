import tensorflow  as tf
from tensorflow.keras import layers,Model

def _depthwise_separable_conv_block(inputs,filters,stride,name):
  x=layers.DepthwiseConv2D(
      kernel_size=(3,3),
      strides=stride,
      padding='same',
      use_bias=False,
      name=f"{name}_depthwise_conv" # Changed / to _
  )(inputs)
  x=layers.BatchNormalization(name=f'{name}_depthwise_bn')(x) # Changed / to _
  x=layers.ReLU(6.,name=f'{name}_depthwise_relu')(x) # Changed / to _
  x=layers.Conv2D(
      filters,
      kernel_size=(1,1),
      strides=(1,1),
      padding='same',
      use_bias=False,
      name=f'{name}_pointwise_conv' # Changed / to _
  )(x)
  x=layers.BatchNormalization(name=f'{name}_pointwise_bn')(x) # Changed / to _
  return layers.ReLU(6.,name=f'{name}_pointwise_relu')(x) # Changed / to _