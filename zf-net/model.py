import tensorflow as tf
from tensorflow.keras import layers,models

def build_zfnet_advanced(input_shape=(224,224,3),num_classes=1000):
  model=models.Sequential([
      layers.Conv2D(96,(7,7),strides=(2,2),activation='relu',input_shape=input_shape,
                    kernel_initializer='he_normal',padding='valid'),
      layers.BatchNormalization(),
      layers.MaxPooling2D((3,3),strides=(2,2),padding='valid'),
      layers.Conv2D(256,(5,5),strides=(2,2),activation='relu',kernel_initializer='he_normal',padding='valid'),
      layers.BatchNormalization(),
      layers.MaxPooling2D((3,3),strides=(2,2),padding='valid'),
      layers.Conv2D(512,(3,3),strides=(1,1),activation='relu',kernel_initializer='he_normal',padding='same'),
      layers.BatchNormalization(),
      layers.Conv2D(1024,(3,3),strides=(1,1),activation='relu',kernel_initializer='he_normal',padding='same'),
      layers.BatchNormalization(),
      layers.Conv2D(512,(3,3),strides=(1,1),activation='relu',kernel_initializer='he_normal',padding='same'),
      layers.BatchNormalization(),
      layers.MaxPooling2D((3,3),strides=(2,2),padding='valid'),
      layers.Flatten(),
      layers.Dense(4096,activation='relu',kernel_initializer='he_normal'),
      layers.Dropout(0.5),
      layers.Dense(4096,activation='relu',kernel_initializer='he_normal'),
      layers.Dropout(0.5),
      layers.Dense(num_classes,activation='softmax')
  ])
  return model