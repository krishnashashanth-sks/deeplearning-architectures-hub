from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, PReLU, BatchNormalization, Add, Concatenate
from layers import conv_bLock_3d

def build_vnet(input_shape,num_classes,base_n_filters=16):
  inputs=Input(shape=input_shape)

  # Encoder Path
  # Level 1
  conv1=conv_block_3d(inputs,base_n_filters)
  down1=conv_block_3d(conv1,base_n_filters*2,strides=(2,2,2),residual=False)

  # Level 2
  conv2=conv_block_3d(down1,base_n_filters*2) # Corrected: input should be down1
  down2=conv_block_3d(conv2,base_n_filters*4,strides=(2,2,2),residual=False)

  # Level 3
  conv3=conv_block_3d(down2,base_n_filters*4) # Corrected: input should be down2
  down3=conv_block_3d(conv3,base_n_filters*8,strides=(2,2,2),residual=False)

  # Level 4
  conv4=conv_block_3d(down3,base_n_filters*8) # Corrected: input should be down3
  down4=conv_block_3d(conv4,base_n_filters*16,strides=(2,2,2),residual=False)

  # Bottleneck
  bottleneck=conv_block_3d(down4,base_n_filters*16)

  # Decoder Path
  # Level 4
  up4=Conv3DTranspose(base_n_filters*8,kernel_size=(2,2,2),strides=(2,2,2),padding='same')(bottleneck)
  up4=Concatenate()([up4,conv4])
  conv_up4=conv_block_3d(up4,base_n_filters*8)

  # Level 3
  up3=Conv3DTranspose(base_n_filters*4,kernel_size=(2,2,2),strides=(2,2,2),padding='same')(conv_up4)
  up3=Concatenate()([up3,conv3])
  conv_up3=conv_block_3d(up3,base_n_filters*4)

  # Level 2
  up2=Conv3DTranspose(base_n_filters*2,kernel_size=(2,2,2),strides=(2,2,2),padding='same')(conv_up3)
  up2=Concatenate()([up2,conv2])
  conv_up2=conv_block_3d(up2,base_n_filters*2)

  # Level 1
  up1=Conv3DTranspose(base_n_filters,kernel_size=(2,2,2),strides=(2,2,2),padding='same')(conv_up2) # Corrected: filters should be base_n_filters
  up1=Concatenate()([up1,conv1])
  conv_up1=conv_block_3d(up1,base_n_filters) # Corrected: filters should be base_n_filters

  outputs=Conv3D(num_classes,kernel_size=(1,1,1),activation='sigmoid',padding='same')(conv_up1)
  return Model(inputs=[inputs],outputs=[outputs])