from tensorflow import keras
import tensorflow as tf

class ContrastiveLoss(keras.losses.Loss):
  def __init__(self,temperature=0.07,name='contrastive_loss',**kwargs):
    super(ContrastiveLoss,self).__init__(name=name,**kwargs)
    self.temperature=temperature
  def call(self,image_embeddings,text_embeddings):
    # Add a small epsilon before normalization to prevent NaNs if an embedding vector is all zeros
    image_embeddings = image_embeddings + tf.keras.backend.epsilon()
    text_embeddings = text_embeddings + tf.keras.backend.epsilon()

    image_embeddings=tf.math.l2_normalize(image_embeddings,axis=1)
    text_embeddings=tf.math.l2_normalize(text_embeddings,axis=1);
    similarity=tf.matmul(image_embeddings,text_embeddings,transpose_b=True)
    similarity=similarity/self.temperature
    labels=tf.eye(tf.shape(similarity)[0])
    # Corrected function call to tf.keras.losses.categorical_crossentropy
    loss_it=tf.keras.losses.categorical_crossentropy(
        labels,similarity,from_logits=True
    )
    # Corrected function call to tf.keras.losses.categorical_crossentropy
    loss_ti=tf.keras.losses.categorical_crossentropy(
        tf.transpose(labels),similarity,from_logits=True
    )
    total_loss=(loss_it+loss_ti)/2
    return total_loss