import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DropPath(keras.layers.Layer):
  def __init__(self, drop_prob=None, **kwargs):
    super(DropPath, self).__init__(**kwargs)
    self.drop_prob = drop_prob

  def call(self, inputs, training=None):
    # Only apply drop path during training and if drop_prob > 0
    if training and self.drop_prob is not None and self.drop_prob > 0.0:
      keep_prob = 1.0 - self.drop_prob
      # Construct mask_shape using only TensorFlow operations
      batch_size = tf.shape(inputs)[0]
      # Create a tensor of ones for the non-batch dimensions, e.g., [1, 1, 1] for a 4D tensor
      ones_for_broadcast = tf.ones(tf.rank(inputs) - 1, dtype=tf.int32)
      # Concatenate batch_size and ones to form the full mask shape: [batch_size, 1, 1, 1]
      mask_shape = tf.concat([tf.expand_dims(batch_size, axis=0), ones_for_broadcast], axis=0)

      random_tensor = tf.random.uniform(mask_shape, dtype=inputs.dtype)
      # Create binary mask: 1 if keep, 0 if drop
      binary_tensor = tf.cast(random_tensor < keep_prob, inputs.dtype)
      # Scale the output by 1/keep_prob to maintain expected magnitude
      output = inputs * binary_tensor / keep_prob
      return output
    return inputs # Return inputs directly if not training or drop_prob is 0

  def get_config(self):
    config = super(DropPath, self).get_config()
    config.update({"drop_prob": self.drop_prob})
    return config

class FractalBlock(keras.layers.Layer):
  def __init__(self,filters,num_columns,drop_prob,**kwargs):
    super(FractalBlock,self).__init__(**kwargs)
    self.filters=filters
    self.num_columns=num_columns
    self.drop_prob=drop_prob
    self.conv_blocks_per_column=[]
    for c in range(self.num_columns):
      column_blocks=[]
      for _ in range(c+1):
        column_blocks.append(self._conv_block())
      self.conv_blocks_per_column.append(column_blocks)
    self.drop_path=DropPath(drop_prob=self.drop_prob)
    self.projection = None # Initialize projection layer

  def build(self, input_shape):
    # Create a projection layer if input channels do not match the block's filters
    if input_shape[-1] != self.filters:
      self.projection = layers.Conv2D(self.filters, kernel_size=1, padding='same', name="residual_projection")
    super(FractalBlock, self).build(input_shape)

  def _conv_block(self):
    return keras.Sequential([
        layers.Conv2D(self.filters,kernel_size=3,padding='same'),
        layers.BatchNormalization(),
        layers.ReLU()
    ])

  def call(self,inputs,training=None):
    all_paths_output=[]
    for c in range(self.num_columns):
      x=inputs
      for conv_block in self.conv_blocks_per_column[c]:
        x=conv_block(x)
      all_paths_output.append(x)
    summed_output=tf.add_n(all_paths_output)

    # Apply projection to inputs if necessary for residual connection
    if self.projection:
        inputs_for_residual = self.projection(inputs)
    else:
        inputs_for_residual = inputs

    output=self.drop_path(inputs_for_residual + summed_output,training=training)
    return output

  def get_config(self):
        config = super(FractalBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "num_columns": self.num_columns,
                "drop_prob": self.drop_prob,
            }
        )
        return config