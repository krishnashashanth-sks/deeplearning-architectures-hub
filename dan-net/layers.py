from tensorflow.keras import layers

def residual_block(x,filters,kernel_size=3,stride=1,activation='relu',use_bias=False):
  shortcut=x
  x=layers.Conv2D(filters,kernel_size,strides=stride,padding='same',use_bias=use_bias)(x)
  x=layers.BatchNormalization()(x)
  x=layers.Activation(activation)(x)
  x=layers.Conv2D(filters,kernel_size,padding='same',use_bias=use_bias)(x)
  if stride!=1 or shortcut.shape[-1]!=filters:
    shortcut=layers.Conv2D(filters,1,strides=stride,use_bias=use_bias)(shortcut)
    shortcut=layers.BatchNormalization()(shortcut)
  x=layers.add([shortcut,x])
  x=layers.Activation(activation)(x)
  return x