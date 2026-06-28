from tensorflow.keras.layers import Layer,  Dense, Concatenate, Activation, BatchNormalization
import tensorflow as tf

class LocalActivationUnit(Layer):
  def __init__(self,hidden_units=(36,1),activation='sigmoid',**kwargs):
    super(LocalActivationUnit,self).__init__(**kwargs)
    self.hidden_units=hidden_units
    self.activation=activation
    self.dense_layers=[]
  def build(self,input_shape):
    input_dim=input_shape[-1]
    for i,units in enumerate(self.hidden_units):
      self.dense_layers.append(Dense(units,activation='relu' if i <len(self.hidden_units)-1 else None) )
    self.output_activation=Activation(self.activation)
    super(LocalActivationUnit,self).build(input_shape)
  def call(self,inputs,**kwargs):
    query_emb,key_embs=inputs
    seq_len=tf.shape(key_embs)[1]
    embedding_dim=tf.shape(query_emb)[-1]
    query_emb_expanded=tf.expand_dims(query_emb,axis=1)
    query_emb_repeated=tf.tile(query_emb_expanded,[1,seq_len,1])
    concat_features=Concatenate(axis=-1)([
        query_emb_repeated,
        key_embs,
        query_emb_repeated-key_embs,
        query_emb_repeated*key_embs
    ])
    attention_scores=concat_features
    for dense_layer in self.dense_layers:
      attention_scores=dense_layer(attention_scores)
    attention_scores=tf.squeeze(attention_scores,axis=-1)
    attention_weights=self.output_activation(attention_scores)
    attention_weights=tf.expand_dims(attention_weights,axis=-1)
    weighted_hist_embs=key_embs*attention_weights
    dynamic_user_interest=tf.reduce_sum(weighted_hist_embs,axis=1)
    return dynamic_user_interest