from tensorflow.keras import layers,models

def build_vggish_keras_model(input_shape=(96,64,1),embedding_dim=128):
  model=models.Sequential(name="vggish")
  model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same',input_shape=input_shape,name='conv1'))
  model.add(layers.MaxPool2D((2,2),strides=(2,2),name='pool1'))

  model.add(layers.Conv2D(128,(3,3),activation='relu',padding='same',input_shape=input_shape,name='conv2'))
  model.add(layers.MaxPool2D((2,2),strides=(2,2),name='pool2'))

  model.add(layers.Conv2D(256,(3,3),activation='relu',padding='same',input_shape=input_shape,name='conv3_conv3_1'))
  model.add(layers.Conv2D(256,(3,3),activation='relu',padding='same',input_shape=input_shape,name='conv3_conv3_2'))
  model.add(layers.MaxPool2D((2,2),strides=(2,2),name='pool3'))

  model.add(layers.Conv2D(512,(3,3),activation='relu',padding='same',input_shape=input_shape,name='conv4_conv4_1'))
  model.add(layers.Conv2D(512,(3,3),activation='relu',padding='same',input_shape=input_shape,name='conv4_conv4_2'))
  model.add(layers.MaxPool2D((2,2),strides=(2,2),name='pool4'))

  model.add(layers.Flatten(name='flatten'))
  model.add(layers.Dense(4096,activation='relu',name='fc1'))
  model.add(layers.Dense(4096,activation='relu',name='fc2'))
  model.add(layers.Dense(embedding_dim,activation=None,name='vggish_embeddings'))
  return model

def build_vggish_classfication_model(input_shape=(96, 64, 1),num_classes=10):
    input_tensor = layers.Input(shape=input_shape, name='input_mel_spectrogram')
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1')(input_tensor)
    x = layers.MaxPool2D((2, 2), strides=(2, 2), name='pool1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPool2D((2, 2), strides=(2, 2), name='pool2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_conv3_1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_conv3_2')(x)
    x = layers.MaxPool2D((2, 2), strides=(2, 2), name='pool3')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_conv4_1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_conv4_2')(x)
    x = layers.MaxPool2D((2, 2), strides=(2, 2), name='pool4')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    classification_output = layers.Dense(num_classes, activation='softmax', name='classification_output')(x)

    return models.Model(inputs=input_tensor, outputs=classification_output, name="vggish_classifier")
