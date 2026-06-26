import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,Activation,MaxPooling2D,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def build_segnet(input_shape,num_classes):
  inputs=Input(shape=input_shape)
  conv1_1=Conv2D(64,(3,3),padding='same',name="conv1_1")(inputs)
  conv1_1=BatchNormalization(name="bn1_1")(conv1_1)
  conv1_1=Activation('relu',name='relu1_1')(conv1_1)
  conv1_2=Conv2D(64,(3,3),padding='same',name='conv1_2')(conv1_1)
  conv1_2=BatchNormalization(name='bn1_2')(conv1_2)
  conv1_2=Activation('relu',name='relu1_2')(conv1_2)
  pool1=MaxPooling2D((2,2),strides=(2,2),name='pool1')(conv1_2)

  conv2_1=Conv2D(128,(3,3),padding='same',name="conv2_1")(pool1) # Corrected input and filters
  conv2_1=BatchNormalization(name="bn2_1")(conv2_1)
  conv2_1=Activation('relu',name='relu2_1')(conv2_1)
  conv2_2=Conv2D(128,(3,3),padding='same',name='conv2_2')(conv2_1)
  conv2_2=BatchNormalization(name='bn2_2')(conv2_2)
  conv2_2=Activation('relu',name='relu2_2')(conv2_2)
  pool2=MaxPooling2D((2,2),strides=(2,2),name='pool2')(conv2_2) # Corrected name

  conv3_1=Conv2D(256,(3,3),padding='same',name="conv3_1")(pool2)
  conv3_1=BatchNormalization(name="bn3_1")(conv3_1)
  conv3_1=Activation('relu',name='relu3_1')(conv3_1)
  conv3_2=Conv2D(256,(3,3),padding='same',name='conv3_2')(conv3_1)
  conv3_2=BatchNormalization(name='bn3_2')(conv3_2)
  conv3_2=Activation('relu',name='relu3_2')(conv3_2)
  conv3_3=Conv2D(256,(3,3),padding='same',name='conv3_3')(conv3_2)
  conv3_3=BatchNormalization(name='bn3_3')(conv3_3)
  conv3_3=Activation('relu',name='relu3_3')(conv3_3)
  pool3=MaxPooling2D((2,2),strides=(2,2),name='pool3')(conv3_3) # Corrected name

  conv4_1=Conv2D(512,(3,3),padding='same',name="conv4_1")(pool3)
  conv4_1=BatchNormalization(name="bn4_1")(conv4_1)
  conv4_1=Activation('relu',name='relu4_1')(conv4_1)
  conv4_2=Conv2D(512,(3,3),padding='same',name='conv4_2')(conv4_1)
  conv4_2=BatchNormalization(name='bn4_2')(conv4_2)
  conv4_2=Activation('relu',name='relu4_2')(conv4_2)
  conv4_3=Conv2D(512,(3,3),padding='same',name='conv4_3')(conv4_2)
  conv4_3=BatchNormalization(name='bn4_3')(conv4_3)
  conv4_3=Activation('relu',name='relu4_3')(conv4_3)
  pool4=MaxPooling2D((2,2),strides=(2,2),name='pool4')(conv4_3) # Corrected name

  conv5_1=Conv2D(512,(3,3),padding='same',name="conv5_1")(pool4)
  conv5_1=BatchNormalization(name="bn5_1")(conv5_1)
  conv5_1=Activation('relu',name='relu5_1')(conv5_1)
  conv5_2=Conv2D(512,(3,3),padding='same',name='conv5_2')(conv5_1)
  conv5_2=BatchNormalization(name='bn5_2')(conv5_2)
  conv5_2=Activation('relu',name='relu5_2')(conv5_2)
  conv5_3=Conv2D(512,(3,3),padding='same',name='conv5_3')(conv5_2)
  conv5_3=BatchNormalization(name='bn5_3')(conv5_3)
  conv5_3=Activation('relu',name='relu5_3')(conv5_3)
  pool5=MaxPooling2D((2,2),strides=(2,2),name='pool5')(conv5_3) # Corrected name

  #Decoder
  uppool5=UpSampling2D((2,2),name='uppool_5')(pool5)
  deconv5_3=Conv2D(512,(3,3),padding='same',name="deconv5_3")(uppool5)
  deconv5_3=BatchNormalization(name="debn5_3")(deconv5_3)
  deconv5_3=Activation('relu',name='derelu5_3')(deconv5_3)
  deconv5_2=Conv2D(512,(3,3),padding='same',name='deconv5_2')(deconv5_3)
  deconv5_2=BatchNormalization(name='debn5_2')(deconv5_2)
  deconv5_2=Activation('relu',name='derelu5_2')(deconv5_2)
  deconv5_1=Conv2D(512,(3,3),padding='same',name='deconv5_1')(deconv5_2)
  deconv5_1=BatchNormalization(name='debn5_1')(deconv5_1) # Corrected name
  deconv5_1=Activation('relu',name='derelu5_1')(deconv5_1) # Corrected name

  uppool4=UpSampling2D((2,2),name='uppool_4')(deconv5_1)
  deconv4_3=Conv2D(512,(3,3),padding='same',name="deconv4_3")(uppool4)
  deconv4_3=BatchNormalization(name="debn4_3")(deconv4_3)
  deconv4_3=Activation('relu',name='derelu4_3')(deconv4_3)
  deconv4_2=Conv2D(512,(3,3),padding='same',name='deconv4_2')(deconv4_3)
  deconv4_2=BatchNormalization(name='debn4_2')(deconv4_2)
  deconv4_2=Activation('relu',name='derelu4_2')(deconv4_2)
  deconv4_1=Conv2D(256,(3,3),padding='same',name='deconv4_1')(deconv4_2) # Corrected filter count
  deconv4_1=BatchNormalization(name='debn4_1')(deconv4_1)
  deconv4_1=Activation('relu',name='derelu4_1')(deconv4_1) # Corrected name

  uppool3=UpSampling2D((2,2),name='uppool_3')(deconv4_1)
  deconv3_3=Conv2D(256,(3,3),padding='same',name="deconv3_3")(uppool3) # Corrected filter count
  deconv3_3=BatchNormalization(name="debn3_3")(deconv3_3)
  deconv3_3=Activation('relu',name='derelu3_3')(deconv3_3)
  deconv3_2=Conv2D(256,(3,3),padding='same',name='deconv3_2')(deconv3_3) # Corrected filter count
  deconv3_2=BatchNormalization(name='debn3_2')(deconv3_2)
  deconv3_2=Activation('relu',name='derelu3_2')(deconv3_2)
  deconv3_1=Conv2D(128,(3,3),padding='same',name='deconv3_1')(deconv3_2) # Corrected filter count
  deconv3_1=BatchNormalization(name='debn3_1')(deconv3_1) # Corrected name
  deconv3_1=Activation('relu',name='derelu3_1')(deconv3_1) # Corrected name

  uppool2=UpSampling2D((2,2),name='uppool_2')(deconv3_1)
  deconv2_2=Conv2D(128,(3,3),padding='same',name='deconv2_2')(uppool2) # Corrected filter count
  deconv2_2=BatchNormalization(name='debn2_2')(deconv2_2)
  deconv2_2=Activation('relu',name='derelu2_2')(deconv2_2)
  deconv2_1=Conv2D(64,(3,3),padding='same',name='deconv2_1')(deconv2_2) # Corrected filter count
  deconv2_1=BatchNormalization(name='debn2_1_dec')(deconv2_1) # Corrected name for uniqueness
  deconv2_1=Activation('relu',name='derelu2_1')(deconv2_1) # Corrected name

  uppool1=UpSampling2D((2,2),name='uppool_1')(deconv2_1)
  deconv1_2=Conv2D(64,(3,3),padding='same',name='deconv1_2')(uppool1) # Corrected filter count
  deconv1_2=BatchNormalization(name='debn1_2')(deconv1_2)
  deconv1_2=Activation('relu',name='derelu1_2')(deconv1_2)
  deconv1_1=Conv2D(64,(3,3),padding='same',name='deconv1_1')(deconv1_2) # Corrected filter count
  deconv1_1=BatchNormalization(name='debn1_1_dec')(deconv1_1) # Corrected name for uniqueness
  deconv1_1=Activation('relu',name='derelu1_1')(deconv1_1) # Corrected name

  outputs=Conv2D(num_classes,(1,1),activation='softmax',padding='same',name='outputs')(deconv1_1)
  model=Model(inputs=inputs,outputs=outputs,name='SegNet')
  return model