from tensorflow.keras import layers

def depthwise_separable_conv_block(inputs,filters,kernel_size=(3,3),strides=(1,1),use_bn=True,activation='relu',block_id=None):
  prefix='block'+str(block_id)+'_'
  x=layers.DepthwiseConv2D(
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      name=prefix+'depthwise_conv'
  )(inputs)
  if use_bn:
    x=layers.BatchNormalization(name=prefix+'depthwise_bn')(x)
  if activation is not None:
    x=layers.Activation(activation,name=prefix+'depthwise_act')(x)
  x=layers.Conv2D(
      filters,
      (1,1),
      padding='same',
      use_bias=False,
      name=prefix+'pointwise_conv'
  )(x)
  if use_bn:
    x=layers.BatchNormalization(name=prefix+'pointwise_bn')(x)
  if activation is not None:
    x=layers.Activation(activation,name=prefix+'pointwise_act')(x) # Corrected: added (x) to apply the activation
  return x

def xception_entry_flow(inputs):
  x=layers.Conv2D(32,(3,3),strides=(2,2),use_bias=False,name='block1_conv1')(inputs)
  x=layers.BatchNormalization(name='block1_conv1_bn')(x)
  x=layers.Activation('relu',name='block1_conv1_act')(x)
  x=layers.Conv2D(64,(3,3),use_bias=False,name='block1_conv2')(x)
  x=layers.BatchNormalization(name='block1_conv2_bn')(x)
  x=layers.Activation('relu',name='block1_conv2_act')(x)

  residual=layers.Conv2D(128,(1,1),strides=(2,2),padding='same',use_bias=False,name='block2_res_conv')(x)
  residual=layers.BatchNormalization(name='block2_res_bn')(residual)
  x=depthwise_separable_conv_block(x,128,block_id=2,activation=None)
  x=layers.MaxPooling2D((3,3),strides=(2,2),padding='same',name='block2_pool')(x)
  x=layers.add([x,residual],name='block2_add')
  x=layers.Activation('relu',name='block2_act')(x)

  residual=layers.Conv2D(256,(1,1),strides=(2,2),padding='same',use_bias=False,name='block3_res_conv')(x)
  residual=layers.BatchNormalization(name='block3_res_bn')(residual)
  x=depthwise_separable_conv_block(x,256,block_id=3,activation=None)
  x=layers.MaxPooling2D((3,3),strides=(2,2),padding='same',name='block3_pool')(x)
  x=layers.add([x,residual],name='block3_add')
  x=layers.Activation('relu',name='block3_act')(x)

  residual=layers.Conv2D(728,(1,1),strides=(2,2),padding='same',use_bias=False,name='block4_res_conv')(x)
  residual=layers.BatchNormalization(name='block4_res_bn')(residual)
  x=depthwise_separable_conv_block(x,728,block_id=4,activation=None)
  x=layers.MaxPooling2D((3,3),strides=(2,2),padding='same',name='block4_pool')(x)
  x=layers.add([x,residual],name='block4_add')
  x=layers.Activation('relu',name='block4_act')(x)
  return x

def xception_middle_flow(inputs):
  x=inputs
  for i in range(8):
    residual=x
    x=depthwise_separable_conv_block(x,728,block_id=5+i,activation=None)
    x=layers.add([x,residual],name=f'block{5+i}_add')
    x=layers.Activation('relu',name=f'block{5+i}_act')(x)
  return x

def xception_exit_flow(inputs,num_classes):
  residual=layers.Conv2D(1024,(1,1),strides=(2,2),padding='same',use_bias=False,name='block13_res_conv')(inputs)
  residual=layers.BatchNormalization(name='block13_res_bn')(residual)
  x=depthwise_separable_conv_block(inputs,728,activation=None,block_id=13)
  x=depthwise_separable_conv_block(x,1024,activation=None,block_id=14) # Changed filters from 728 to 1024 to match residual
  x=layers.MaxPooling2D((3,3),strides=(2,2),padding='same',name='block13_pool')(x)
  x=layers.add([x,residual],name='block13_add')
  x=layers.Activation('relu',name='block13_act')(x)
  x=depthwise_separable_conv_block(x,1536,block_id=15)
  x=depthwise_separable_conv_block(x,2048,block_id=16)
  x=layers.GlobalAveragePooling2D(name='avg_pool')(x)
  outputs=layers.Dense(num_classes,activation='softmax',name='predictions')(x)
  return outputs