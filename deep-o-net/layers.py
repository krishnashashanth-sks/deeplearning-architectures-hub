import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
class BranchNetwork(keras.Model):
  def __init__(self,output_dim,input_shape):
    super(BranchNetwork,self).__init__()
    self.dense1=layers.Dense(128,activation='relu',input_shape=(input_shape,))
    self.batch_norm1=layers.BatchNormalization()
    self.dense2=layers.Dense(64,activation='relu')
    self.batch_norm2=layers.BatchNormalization()
    self.output_layer=layers.Dense(output_dim)
  def call(self,inputs):
    return self.output_layer(self.batch_norm2(self.dense2(self.batch_norm1(self.dense1(inputs)))))

class TrunkNetwork(keras.Model):
  def __init__(self,output_dim,input_shape):
    super(TrunkNetwork,self).__init__()
    self.dense1=layers.Dense(128,activation='relu',input_shape=(input_shape,))
    self.batch_norm1=layers.BatchNormalization()
    self.dense2=layers.Dense(64,activation='relu')
    self.batch_norm2=layers.BatchNormalization()
    self.output_layer=layers.Dense(output_dim)
  def call(self,inputs):
    return self.output_layer(self.batch_norm2(self.dense2(self.batch_norm1(self.dense1(inputs)))))