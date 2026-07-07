from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

class ResidualBlock(layers.Layer):
  def __init__(self,filters,kernel_size,stride=1,activation='relu',**kwargs):
    super(ResidualBlock,self).__init__(**kwargs)
    self.conv1=layers.Conv2D(filters,kernel_size,strides=stride,padding='same',use_bias=False)
    self.bn1=layers.BatchNormalization()
    self.activation=layers.Activation(activation)
    self.conv2=layers.Conv2D(filters,kernel_size,strides=1,padding='same',use_bias=False)
    self.bn2=layers.BatchNormalization()
    if stride !=1 or filters!=filters: # The second part of this condition `filters!=filters` will always be false. It should probably be `filters != input_filters` or removed.
      self.shortcut=keras.Sequential([
          layers.Conv2D(filters,1,strides=stride,use_bias=False),
          layers.BatchNormalization()
      ])
    else:
      self.shortcut=tf.identity
  def call(self,inputs):
    residual=self.shortcut(inputs)
    x=self.bn2(self.conv2(self.activation(self.bn1(self.conv1(inputs)))))
    return self.activation(layers.add([x,residual]))

class ImageEncoder(keras.Model):
  def __init__(self,embedding_dim,**kwargs):
    super(ImageEncoder,self).__init__(**kwargs)
    self.conv_initial=layers.Conv2D(64,(7,7),strides=(2,2),padding='same',use_bias=False)
    self.bn_initia=layers.BatchNormalization()
    self.activation=layers.ReLU()
    self.max_pool=layers.MaxPooling2D((3,3),strides=(2,2),padding='same')
    # Define distinct residual blocks
    self.res_block1=ResidualBlock(64,(3,3))
    self.res_block2=ResidualBlock(128,(3,3),stride=2)
    self.res_block3=ResidualBlock(256,(3,3),stride=2)
    self.res_block4=ResidualBlock(512,(3,3),stride=2)
    self.global_avg_pool=layers.GlobalAveragePooling2D() 
    self.dense_projection=layers.Dense(embedding_dim,activation='relu')

  def call(self,inputs):
    x=self.conv_initial(inputs)
    x=self.bn_initia(x) 
    x=self.activation(x)
    x=self.max_pool(x)
    # Call each residual block sequentially
    x=self.res_block1(x)
    x=self.res_block2(x)
    x=self.res_block3(x)
    x=self.res_block4(x)
    attention_features=x
    x=self.global_avg_pool(x)
    image_embedding=self.dense_projection(x)
    return image_embedding,attention_features

class BahdanauAttention(layers.Layer):
  def __init__(self,units,**kwargs):
    super(BahdanauAttention,self).__init__(**kwargs)
    self.W1=layers.Dense(units)
    self.W2=layers.Dense(units)
    self.V=layers.Dense(1)
  def call(self,features,hidden):
    hidden_with_time_axis=tf.expand_dims(tf.expand_dims(hidden,1),2)
    scores=tf.nn.tanh(self.W1(features)+self.W2(hidden_with_time_axis))
    attention_logits=self.V(scores)
    score_shape=tf.shape(attention_logits)
    batch_size=score_shape[0]
    H=score_shape[1]
    W=score_shape[2]
    reshaped_logits=tf.reshape(attention_logits,(batch_size,H*W,1))
    softmax_result=tf.nn.softmax(reshaped_logits,axis=1)
    attention_weights=tf.reshape(softmax_result,(batch_size,H,W,1))
    context_vector=attention_weights*features
    context_vector=tf.reduce_mean(context_vector,axis=(1,2))
    return context_vector,attention_weights

class TextDecoder(keras.Model):
  def __init__(self,embedding_dim,units,vocab_size,**kwargs):
    super(TextDecoder,self).__init__(**kwargs)
    self.units=units
    self.vocab_size=vocab_size
    self.embedding=layers.Embedding(self.vocab_size,embedding_dim)
    self.lstm=layers.LSTM(self.units,
                          return_sequences=True,
                          return_state=True,
                          recurrent_initializer='glorot_uniform')
    self.fc1=layers.Dense(self.units)
    self.fc2=layers.Dense(self.vocab_size)
    self.attention=BahdanauAttention(self.units)
  def call(self,x,features,hidden):
    context_vector,attention_weights=self.attention(features,hidden)
    x=self.embedding(x)
    context_vector_expanded=tf.expand_dims(context_vector,1)
    x=tf.concat([context_vector_expanded,x],axis=-1)
    output,state_h,state_c=self.lstm(x)
    x=self.fc1(output)
    x=tf.reshape(x,(-1,x.shape[2]))
    x=self.fc2(x)
    return x,state_h,state_c,attention_weights
  def reset_state(self,batch_size):
    return tf.zeros((batch_size,self.units)),tf.zeros((batch_size,self.units))