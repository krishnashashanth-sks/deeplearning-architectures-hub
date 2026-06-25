from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, PReLU, BatchNormalization, Add, Concatenate

def conv_block_3d(input_tensor,n_filters,kernel_size=(3,3,3),strides=(1,1,1),activation=True,residual=True):
    shortcut=input_tensor
    x=Conv3D(filters=n_filters,kernel_size=kernel_size,padding='same',strides=strides)(input_tensor)
    x=BatchNormalization()(x)
    if activation:
      x=PReLU()(x)
    x=Conv3D(filters=n_filters,kernel_size=kernel_size,padding='same')(x)
    x=BatchNormalization()(x)
    if residual:
      if shortcut.shape[-1]!=n_filters or strides!=(1,1,1):
        shortcut=Conv3D(filters=n_filters,kernel_size=(1,1,1),strides=strides,padding='same')(shortcut)
        shortcut=BatchNormalization()(shortcut)
      x=Add()([x,shortcut])
    if activation:
      x=PReLU()(x)
    return x