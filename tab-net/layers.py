import tensorflow as tf
from tensorflow.keras import layers

class GLU(layers.Layer):
  def __init__(self,units,**kwargs):
    super().__init__(**kwargs)
    self.linear=layers.Dense(units)
    self.gate=layers.Dense(units,activation='sigmoid')

  def build(self, input_shape):
    self.linear.build(input_shape)
    self.gate.build(input_shape)
    super().build(input_shape)

  def call(self,inputs):
    return self.linear(inputs)*self.gate(inputs)
  
class GBN(layers.Layer):
    def __init__(self, virtual_batch_size, momentum=0.9, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        # Setting axis=-1 allows normalization strictly across the features,
        # regardless of whether the input is 2D or 3D.
        self.bn = layers.BatchNormalization(
            axis=-1,
            momentum=self.momentum,
            epsilon=self.epsilon
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        if training:
            batch_size = tf.shape(inputs)[0] # Corrected: Get the scalar batch size
            features_dim = inputs.shape[-1]

            # Dynamic check: Is the batch perfectly divisible by the virtual size?
            is_divisible = tf.equal(tf.math.mod(batch_size, self.virtual_batch_size), 0)
            is_large_enough = tf.greater_equal(batch_size, self.virtual_batch_size)

            def GhostBN():
                # Shape: (num_virtual_batches, virtual_batch_size, features) -> ndim=3
                inputs_reshaped = tf.reshape(inputs, [-1, self.virtual_batch_size, features_dim])
                outputs = self.bn(inputs_reshaped, training=True)
                return tf.reshape(outputs, tf.shape(inputs))

            def StandardBN():
                # Expand 2D (batch_size, features) to 3D (batch_size, 1, features)
                # This tricks Keras into keeping a uniform ndim=3 across both code paths!
                inputs_expanded = tf.expand_dims(inputs, axis=1)
                outputs = self.bn(inputs_expanded, training=True)
                return tf.reshape(outputs, tf.shape(inputs))

            return tf.cond(tf.logical_and(is_divisible, is_large_enough), GhostBN, StandardBN)
        else:
            return self.bn(inputs, training=False)
        
class FeatureTransformer(layers.Layer):
  def __init__(self,input_dim,output_dim,num_glu_shared_blocks,num_glu_independent_blocks,virtual_batch_size,momentum=0.9,**kwargs):
    super().__init__(**kwargs)
    self.input_dim=input_dim
    self.output_dim=output_dim
    self.num_glu_shared_blocks=num_glu_shared_blocks
    self.num_glu_independent_blocks=num_glu_independent_blocks
    self.virtual_batch_size=virtual_batch_size
    self.momentum=momentum
    self.shared_glu_blocks=[]
    for _ in range(num_glu_shared_blocks):
      self.shared_glu_blocks.append(GBN(virtual_batch_size,momentum=momentum))
      self.shared_glu_blocks.append(GLU(output_dim))
    self.independent_glu_blocks=[]
    for _ in range(num_glu_independent_blocks):
      self.independent_glu_blocks.append(GBN(virtual_batch_size,momentum=momentum))
      self.independent_glu_blocks.append(GLU(output_dim))
    self.initial_dense=layers.Dense(output_dim,use_bias=False)
  def call(self,inputs,training=False):
    x=self.initial_dense(inputs)
    for i,block in enumerate(self.shared_glu_blocks):
      if i%2==0:
        x=block(x,training=training)
      else:
        x=x*tf.math.sqrt(0.5)+block(x)*tf.math.sqrt(0.5)
    for i,block in enumerate(self.independent_glu_blocks):
      if i % 2==0:
        x=block(x,training=training)
      else:
        x=x*tf.math.sqrt(0.5)+block(x)*tf.math.sqrt(0.5)
    return x
  
class AttentiveTransformer(layers.Layer):
  def __init__(self,output_dim,virtual_batch_size,momentum=0.9,epsilon=1e-15,**kwargs):
    super().__init__(**kwargs)
    self.output_dim=output_dim
    self.virtual_batch_size=virtual_batch_size
    self.momentum=momentum
    self.epsilon=epsilon
    self.glu=GLU(output_dim,name='attentive_glu') # Fixed: self=glu -> self.glu
    self.gbn=GBN(virtual_batch_size,momentum=momentum,name='attentive_gbn')
    self.dense=layers.Dense(output_dim,use_bias=False,name='attentive_dense')

  def build(self, input_shape):
    self.glu.build(input_shape)
    self.gbn.build(input_shape)
    # The dense layer's input will be the output of GLU, which has 'output_dim' features.
    # So its input shape will be (batch_size, self.output_dim).
    self.dense.build((input_shape[0], self.output_dim))
    super().build(input_shape)

  def call(self,inputs,prior_scales,training=False):
    x=self.gbn(inputs,training=training)
    x=self.glu(x)
    sparse_weights=tf.nn.softmax(self.dense(x)*prior_scales,axis=-1)
    return sparse_weights