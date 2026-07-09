from tensorflow import keras
from tensorflow.keras import layers

def create_vlm_model(vision_encoder,language_encoder,img_height,img_width,img_channels,sequence_length,embedding_dim):
  image_input=keras.Input(shape=(img_height,img_width,img_channels),name='image_input')
  language_input=keras.Input(shape=(sequence_length,),name='language_input')
  vision_embedding=vision_encoder(image_input)
  language_embedding=language_encoder(language_input)
  fused_embedding=layers.concatenate([vision_embedding,language_embedding],axis=-1,name='fused_embedding')
  return keras.Model(inputs=[image_input,language_input],outputs=fused_embedding,name='vision_language_model')