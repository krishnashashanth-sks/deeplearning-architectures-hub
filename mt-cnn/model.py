import tensorflow as tf

# ---  Model Architectures ---

def build_pnet_fcn(input_shape=(None, None, 3)): # Accepts variable input size
  inputs = tf.keras.Input(shape=input_shape)
  x = tf.keras.layers.Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
  x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')(x)
  x = tf.keras.layers.Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
  x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
  x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
  x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)

  classifier = tf.keras.layers.Conv2D(2, (1, 1), activation='softmax', name='flatten_cls')(x)
  bbox_regressor = tf.keras.layers.Conv2D(4, (1, 1), activation='linear', name='flatten_reg')(x)

  model = tf.keras.models.Model(inputs=inputs, outputs=[classifier, bbox_regressor], name='PNet_FCN')
  return model

def build_rnet(input_shape=(24,24,3)):
  inputs=tf.keras.Input(shape=input_shape)
  x=tf.keras.layers.Conv2D(28,(3,3),strides=1,padding='valid',name='conv1')(inputs)
  x=tf.keras.layers.PReLU(shared_axes=[1,2],name='prelu1')(x)
  x=tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2,name='pool1')(x)
  x=tf.keras.layers.Conv2D(48,(3,3),strides=1,padding='valid',name='conv2')(x)
  x=tf.keras.layers.PReLU(shared_axes=[1,2],name='prelu2')(x)
  x=tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2,name='pool3')(x)
  x=tf.keras.layers.Conv2D(64,(3,3),strides=1,padding='valid',name='conv3')(x)
  x=tf.keras.layers.PReLU(shared_axes=[1,2],name='prelu3')(x)
  x=tf.keras.layers.Flatten(name='flatten')(x)
  x=tf.keras.layers.Dense(128,name='dense1')(x)
  x=tf.keras.layers.PReLU(name='prelu4')(x)
  classifier=tf.keras.layers.Dense(2,activation='softmax',name='cls_output')(x)
  bbox_regressor=tf.keras.layers.Dense(4,activation='linear',name='bbox_output')(x)
  model=tf.keras.models.Model(inputs=inputs,outputs=[classifier,bbox_regressor],name='RNet')
  return model

def build_onet(input_shape=(48,48,3)):
  inputs=tf.keras.Input(shape=input_shape)
  x=tf.keras.layers.Conv2D(32,(3,3),strides=1,padding='valid',name='conv1')(inputs)
  x=tf.keras.layers.PReLU(shared_axes=[1,2],name='prelu1')(x)
  x=tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2,name='pool1')(x)

  x=tf.keras.layers.Conv2D(64,(3,3),strides=1,padding='valid',name='conv2')(x)
  x=tf.keras.layers.PReLU(shared_axes=[1,2],name='prelu2')(x)
  x=tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2,name='pool2')(x)

  x=tf.keras.layers.Conv2D(64,(3,3),strides=1,padding='valid',name='conv3')(x)
  x=tf.keras.layers.PReLU(shared_axes=[1,2],name='prelu3')(x)
  x=tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=2,name='pool3')(x)

  x=tf.keras.layers.Conv2D(128,(2,2),strides=1,padding='valid',name='conv4')(x)
  x=tf.keras.layers.PReLU(shared_axes=[1,2],name='prelu4')(x)

  x=tf.keras.layers.Flatten(name='flatten')(x)
  x=tf.keras.layers.Dense(256,name='dense1')(x)
  x=tf.keras.layers.PReLU(name='prelu5')(x)
  classifier=tf.keras.layers.Dense(2,activation='softmax',name='cls_output')(x)
  bbox_regressor=tf.keras.layers.Dense(4,activation='linear',name='bbox_output')(x)
  landmark_regressor=tf.keras.layers.Dense(10,activation='linear',name='landmark_output')(x)
  model=tf.keras.models.Model(inputs=inputs,outputs=[classifier,bbox_regressor,landmark_regressor],name='ONet')
  return model