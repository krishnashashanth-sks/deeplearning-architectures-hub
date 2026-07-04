from tensorflow import keras
from tensorflow.keras import layers
from layers import AlethiaBlock

def build_alethia_advanced_model(input_shape,num_classes):
  inputs=keras.Input(shape=input_shape)
  x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(inputs)
  x=layers.MaxPooling2D((2,2))(x)
  x=AlethiaBlock(64,(3,3),name='alethais_block_1')(x)
  x=layers.MaxPooling2D((2,2))(x)
  x=AlethiaBlock(128,(3,3),name='alethia_block_2')(x)
  x=layers.MaxPooling2D((2,2))(x)
  x=layers.Flatten()(x)
  x=layers.Dense(256,activation='relu')(x)
  x=layers.Dropout(0.5)(x)
  outputs=layers.Dense(num_classes,activation='softmax')(x)
  model=keras.Model(inputs=inputs,outputs=outputs,name="Alethia_Advanced_Model")
  return model