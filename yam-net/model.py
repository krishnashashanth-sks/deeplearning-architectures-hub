import tensorflow as tf
from tensorflow.keras import layers
from layers import _depthwise_separable_conv_block

def build_yamnet_model(input_shape=(64,96,1),num_classes=521):
  inputs=tf.keras.Input(shape=input_shape,name='input_1')
  x=layers.Conv2D(32,(3,3),strides=(2,2),padding='same',use_bias=False,name='conv1')(inputs)
  x=layers.BatchNormalization(name='conv1_bn')(x) # Fixed: Pass x to BatchNormalization
  x=layers.ReLU(6.,name='conv1_relu')(x) # Fixed: Pass x to ReLU
  x=_depthwise_separable_conv_block(x,64,(1,1),name='block1')
  x=_depthwise_separable_conv_block(x,128,(1,1),name='block2')
  x=_depthwise_separable_conv_block(x,128,(1,1),name='block3')
  x=_depthwise_separable_conv_block(x,256,(1,1),name='block4')
  x=_depthwise_separable_conv_block(x,256,(1,1),name='block5')
  x=_depthwise_separable_conv_block(x,512,(1,1),name='block6')
  for i in range(5):
    x=_depthwise_separable_conv_block(x,512,(1,1),name=f'block{7+i}')
  x=_depthwise_separable_conv_block(x,1024,(1,1),name='block12')
  x=_depthwise_separable_conv_block(x,1024,(1,1),name='block13')
  x=layers.GlobalAveragePooling2D(name='global_average_pooling')(x)
  predictions=layers.Dense(num_classes,activation='sigmoid',name='predictions')(x)
  return tf.keras.Model(inputs=inputs,outputs=predictions,name='yamnet_model')