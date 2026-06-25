from tensorflow.keras import layers
from layers import residual_block
from tensorflow import keras

def build_advanced_dannet(input_shape,num_classes):
  inputs=keras.Input(shape=input_shape)
  x=layers.Conv2D(32,7,strides=2,padding='same',use_bias=False)(inputs)
  x=layers.BatchNormalization()(x)
  x=layers.Activation('relu')(x)
  x=layers.MaxPooling2D(3,strides=2,padding='same')(x)
  x=residual_block(x,filters=64)
  x=residual_block(x,filters=64)
  x=residual_block(x,filters=128,stride=2)
  x=residual_block(x,filters=128)
  x=residual_block(x,filters=256,stride=2)
  x=residual_block(x,filters=256)
  x=layers.GlobalAveragePooling2D()(x)
  x=layers.Dense(num_classes,activation='softmax')(x)
  return keras.Model(inputs,x,name='dannet')