from tensorflow import keras
from losses import ContrastiveLoss
import tensorflow as tf

class ContrastiveCaptioner(keras.Model):
  def __init__(self,image_encoder,text_decoder,tokenizer,max_caption_length,**kwargs):
    super(ContrastiveCaptioner,self).__init__(**kwargs)
    self.image_encoder=image_encoder
    self.text_decoder=text_decoder 
    self.tokenizer=tokenizer
    self.max_caption_length=max_caption_length
    self.loss_tracker=keras.metrics.Mean(name='loss')
    self.contrastive_loss_fn=ContrastiveLoss()
    self.captioning_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
  def compile(self,optimizer,**kwargs):
    super(ContrastiveCaptioner,self).compile(**kwargs)
    self.optimizer=optimizer
  @property
  def metrics(self):
    return [self.loss_tracker]
  def _compute_captioning_loss(self,real_captions,caption_predictions):
    mask=tf.math.logical_not(tf.math.equal(real_captions,self.tokenizer.word_index['<pad>'])) # Corrected logical_nor to logical_not
    loss_=self.captioning_loss_fn(real_captions,caption_predictions)
    mask=tf.cast(mask,dtype=loss_.dtype)
    loss_*=mask
    sum_mask = tf.reduce_sum(mask) # Calculate sum of mask
    if tf.equal(sum_mask, 0): # Check if sum_mask is 0
        return 0.0 # Return 0.0 if no non-padding tokens
    return tf.reduce_sum(loss_)/sum_mask
  def train_step(self,data):
    img_batch,cap_batch=data
    batch_size = tf.shape(img_batch)[0] 
    decoder_hidden,decoder_cell=self.text_decoder.reset_state(batch_size)
    caption_loss=0
    # Use tf.fill to explicitly create the initial dec_input tensor with correct batch_size
    dec_input=tf.fill((batch_size, 1), self.tokenizer.word_index['<start>'])
    with tf.GradientTape() as tape:
      image_embedding,attention_features=self.image_encoder(img_batch)
      for i in tf.range(1,cap_batch.shape[1]):
        predictions,decoder_hidden,decoder_cell,_=self.text_decoder(
            dec_input,attention_features,decoder_hidden
        )
        caption_loss+=self._compute_captioning_loss(cap_batch[:,i],predictions)
        dec_input=tf.expand_dims(cap_batch[:,i],1)
      caption_loss/=tf.cast(cap_batch.shape[1] - 1,tf.float32) # Divide by actual number of steps (excluding <start>)
      text_embedding=decoder_hidden
      contrastive_loss=self.contrastive_loss_fn(image_embedding,text_embedding) # Corrected contrative_loss_fn and image_embedinng
      total_loss=contrastive_loss+caption_loss 
    trainable_variables=self.image_encoder.trainable_variables+self.text_decoder.trainable_variables # Corrected slef.text_decoder.trainable_varaibles
    gradients=tape.gradient(total_loss,trainable_variables)
    self.optimizer.apply_gradients(zip(gradients,trainable_variables))
    self.loss_tracker.update_state(total_loss)
    return {"loss":self.loss_tracker.result()}