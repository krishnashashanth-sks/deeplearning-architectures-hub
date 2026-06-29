from tensorflow import keras
import tensorflow as tf
from layers import *
class TabNet(keras.Model):
  def __init__(self, 
               feature_dim,
               output_dim,
               num_decision_steps,
               relaxation_factor=1.5,
               bn_momentum=0.9,
               virtual_batch_size=256,
               epsilon=1e-15,
               n_d=8,
               n_a=8,
               num_glu_shared_blocks=2,
               num_glu_independent_blocks=2,
               **kwargs):
    super().__init__(**kwargs)
    self.feature_dim=feature_dim
    self.output_dim=output_dim
    self.num_decision_steps=num_decision_steps
    self.relaxation_factor=relaxation_factor
    self.bn_momentum=bn_momentum
    self.virtual_batch_size=virtual_batch_size
    self.epsilon=epsilon
    self.n_d=n_d
    self.n_a=n_a

    self.feature_transformer_shared = []
    for _ in range(num_glu_shared_blocks):
      self.feature_transformer_shared.append(GBN(virtual_batch_size, momentum=bn_momentum))
      self.feature_transformer_shared.append(GLU(self.n_d + self.n_a))

    self.feature_transformer_independent = []
    for _ in range(num_glu_independent_blocks):
      self.feature_transformer_independent.append(GBN(virtual_batch_size, momentum=bn_momentum))
      self.feature_transformer_independent.append(GLU(self.n_d + self.n_a))

    self.attentive_transformers = []
    self.decision_denses = [] # Initialize list for Dense layers
    for _ in range(num_decision_steps):
        self.attentive_transformers.append(AttentiveTransformer(feature_dim, virtual_batch_size, momentum=bn_momentum, epsilon=epsilon))
        self.decision_denses.append(layers.Dense(self.n_d, use_bias=False)) # Instantiate Dense layers here

    self.initial_dense=layers.Dense(self.n_d+self.n_a,use_bias=False)
    self.initial_bn=GBN(virtual_batch_size,momentum=bn_momentum)
    self.final_dense=layers.Dense(output_dim,activation='softmax' if output_dim >1 else 'sigmoid')

  def build(self, input_shape):
    # Build initial dense and BN layers
    self.initial_dense.build(input_shape)
    self.initial_bn.build((input_shape[0], self.n_d + self.n_a))

    # Build shared feature transformer blocks
    current_shape = tf.TensorShape([input_shape[0], self.n_d + self.n_a]) # Use TensorShape for compatibility
    for block in self.feature_transformer_shared:
      block.build(current_shape)

    # Build independent feature transformer blocks
    for block in self.feature_transformer_independent:
      block.build(current_shape)

    # Build attentive transformers. These take a_out (n_a dimension) and prior_scales (feature_dim)
    # The input to the attentive transformer will be (batch_size, self.n_a)
    attentive_input_shape = tf.TensorShape([input_shape[0], self.n_a])
    for block in self.attentive_transformers:
        block.build(attentive_input_shape)

    # Build the decision_denses layers
    for decision_dense in self.decision_denses:
      decision_dense.build(tf.TensorShape([input_shape[0], self.feature_dim])) # Input shape is (batch_size, feature_dim)

    # Build the final dense layer
    self.final_dense.build(tf.TensorShape([input_shape[0], self.n_d]))
    super().build(input_shape)

  def call(self,inputs,training=False):
    batch_size=tf.shape(inputs)[0]
    x=self.initial_dense(inputs)
    x=self.initial_bn(x,training=training)

    aggregated_output = tf.zeros((batch_size, self.n_d))
    total_sparsity_loss = 0.0

    prior_scales = tf.ones((batch_size, self.feature_dim))
    masks=[]

    for i in range(self.num_decision_steps):
      current_feature_output = x
      # Shared blocks
      for j,block in enumerate(self.feature_transformer_shared):
        if j % 2 == 0:
          current_feature_output=block(current_feature_output,training=training)
        else:
          current_feature_output = current_feature_output * tf.math.sqrt(0.5) + block(current_feature_output)*tf.math.sqrt(0.5)

      # Independent blocks
      for j,block in enumerate(self.feature_transformer_independent):
        if j % 2 == 0:
          current_feature_output=block(current_feature_output,training=training)
        else:
          current_feature_output = current_feature_output * tf.math.sqrt(0.5) + block(current_feature_output)*tf.math.sqrt(0.5)

      d_out=current_feature_output[:,:self.n_d]
      a_out=current_feature_output[:,self.n_d:]

      mask=self.attentive_transformers[i](a_out, prior_scales, training=training)
      masks.append(mask)

      masked_features=inputs*mask
      # Use the pre-instantiated Dense layer
      decision_output=self.decision_denses[i](masked_features)

      aggregated_output += decision_output

      prior_scales = prior_scales * (self.relaxation_factor - tf.reduce_sum(mask,axis=1,keepdims=True))

      total_sparsity_loss += tf.reduce_mean(tf.reduce_sum(mask,axis=1)) / self.num_decision_steps

    if training:
      self.add_loss(total_sparsity_loss)

    outputs=self.final_dense(aggregated_output)
    return outputs