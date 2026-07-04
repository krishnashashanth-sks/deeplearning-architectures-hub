from tensorflow.keras import layers
import tensorflow as tf

class AlethiaBlock(layers.Layer):
  def __init__(self,num_filters,kernel_size,strides=1,activation='relu',**kwargs):
    super(AlethiaBlock,self).__init__()
    self.conv1=layers.Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same')
    self.bn1=layers.BatchNormalization()
    self.activation=layers.Activation(activation)
    self.dropout=layers.Dropout(0.3)
    self.avg_pool=layers.GlobalAveragePooling2D()
    self.dense_attn1=layers.Dense(num_filters//8,activation='relu')
    self.dense_attn2=layers.Dense(num_filters,activation='sigmoid')
  def call(self,inputs):
    x=self.dropout(self.activation(self.bn1(self.conv1(inputs))))
    attention=self.dense_attn2(self.dense_attn1(self.avg_pool(x)))
    attention=tf.expand_dims(tf.expand_dims(attention,1),1)
    return x*attention