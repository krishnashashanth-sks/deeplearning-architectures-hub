from tensorflow.keras.layers import  Embedding, Dense, Concatenate,  BatchNormalization
import tensorflow as tf
from tensorflow.keras import Model
from layers import LocalActivationUnit

class DIN(Model):
  def __init__(self,num_users,num_items,num_categories,embedding_dim,seq_len,
               mlp_hidden_units=(200,80,2),attention_hidden_units=(36,1),
               attention_activation='sigmoid',**kwargs):
    super(DIN,self).__init__(**kwargs)
    self.num_users=num_users
    self.num_items=num_items
    self.num_categories=num_categories
    self.embedding_dim=embedding_dim
    self.seq_len=seq_len
    self.user_embedding=Embedding(num_users,embedding_dim,name='user_emb')
    self.item_embedding=Embedding(num_items,embedding_dim,name='item_emb')
    self.category_embedding=Embedding(num_categories,embedding_dim,name='category_emb')
    self.attention_layer=LocalActivationUnit( # Corrected from self.attention_layers
        hidden_units=attention_hidden_units,
        activation=attention_activation,
        name='local_activation_unit'
    )
    self.mlp_layers=[]
    for i,units in enumerate(mlp_hidden_units):
      self.mlp_layers.append(Dense(units,activation='relu',name=f'mlp_dense_{i}'))
      self.mlp_layers.append(BatchNormalization(name=f'mlp_bn_{i}'))
    self.output_layer=Dense(1,activation='sigmoid',name='output_layer')
  def call(self,inputs):
    user_id=inputs['user_id']
    item_id=inputs['item_id']
    category_id=inputs['category_id']
    hist_item_ids=inputs['hist_item_ids'] # Removed trailing comma
    hist_category_ids=inputs['hist_category_ids']
    user_emb=self.user_embedding(user_id)
    item_emb=self.item_embedding(item_id)
    category_emb=self.category_embedding(category_id)
    hist_item_embs=self.item_embedding(hist_item_ids)
    hist_category_embs=self.category_embedding(hist_category_ids) # Changed name for consistency
    dynamic_user_interact=self.attention_layer(
        (item_emb,hist_item_embs)
    )
    concate_features=Concatenate(axis=-1)([user_emb,
                                           item_emb,
                                           category_emb,
                                           dynamic_user_interact])
    x=concate_features
    for layer in self.mlp_layers:
      x=layer(x)
    output=self.output_layer(x)
    return output