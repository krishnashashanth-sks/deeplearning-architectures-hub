import tensorflow as tf
from tensorflow.keras import layers,Model
class SqueezeAndExcite(layers.Layer):
  def __init__(self,filters,reduction_ratio=4,**kwargs):
    super().__init__(**kwargs)
    self.se_reduce=layers.GlobalAveragePooling2D(keepdims=True) # Corrected typo here
    self.se_expand=layers.Conv2D(filters//reduction_ratio,kernel_size=1,activation='relu')
    self.se_project=layers.Conv2D(filters,kernel_size=1,activation='sigmoid')
  def call(self,inputs):
    x=self.se_project(self.se_expand(self.se_reduce(inputs)))
    return inputs*x

class MBConv(layers.Layer):
  def __init__(self,filters_in,filters_out,kernel_size=3,stride=2,expand_ratio=6,se_ratio=0.25,activation='swish',**kwargs):
    super().__init__(**kwargs)
    self.stride=stride
    self.filters_in=filters_in
    self.filter_out=filters_out
    self.expand_ratio=expand_ratio
    self.use_se=se_ratio>0 and se_ratio <=1
    self.activation_fn=activation # Renamed to avoid conflict with layer.Activation
    filters_expanded=filters_in*expand_ratio
    if expand_ratio!=1:
      self.expand_conv=layers.Conv2D(filters_expanded,kernel_size=1,padding='same',use_bias=False)
      self.expand_bn=layers.BatchNormalization()
      self.expand_activation=layers.Activation(activation)
    else:
      self.expand_conv=None
    self.depthwise_conv=layers.DepthwiseConv2D(kernel_size=kernel_size,strides=stride,padding='same',use_bias=False)
    self.depthwise_bn=layers.BatchNormalization()
    self.depthwise_activation=layers.Activation(activation)
    if self.use_se:
      self.se_block=SqueezeAndExcite(filters_expanded,int(1/se_ratio),**kwargs)
    else:
      self.se_block=None
    self.project_conv=layers.Conv2D(filters_out,kernel_size=1,padding='same',use_bias=False)
    self.project_bn=layers.BatchNormalization()
    self.add_skip=(stride==1) and (filters_in==filters_out)

  def call(self,inputs,training=False): # Added training argument
    x=inputs
    if self.expand_conv:
      x=self.expand_activation(self.expand_bn(self.expand_conv(x), training=training)) # Pass training
    x=self.depthwise_activation(self.depthwise_bn(self.depthwise_conv(x), training=training)) # Pass training
    if self.se_block:
      x=self.se_block(x)
    x=self.project_bn(self.project_conv(x), training=training) # Pass training
    if self.add_skip:
      x=layers.add([x,inputs])
    return x

class MultiHeadSelfAttention(layers.Layer):
  def __init__(self,embed_dim,num_heads=8,**kwargs):
    super().__init__(**kwargs)
    self.embed_dim=embed_dim
    self.num_heads=num_heads
    if embed_dim % num_heads !=0:
      raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
    self.proj_dim=embed_dim//num_heads
    self.query_dense=layers.Dense(embed_dim)
    self.key_dense=layers.Dense(embed_dim)
    self.value_dense=layers.Dense(embed_dim)
    self.combined_heads=layers.Dense(embed_dim)

  def attention(self, query, key, value):
      score = tf.matmul(query, key, transpose_b=True)
      dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
      scaled_score = score / tf.math.sqrt(dim_key)
      weights = tf.nn.softmax(scaled_score, axis=-1)
      output = tf.matmul(weights, value)
      return output, weights

  def separate_heads(self, x, batch_size):
      x = tf.reshape(x, (batch_size, -1, self.num_heads, self.proj_dim))
      return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, inputs):
      batch_size = tf.shape(inputs)[0]

      query = self.query_dense(inputs)
      key = self.key_dense(inputs)
      value = self.value_dense(inputs)

      query = self.separate_heads(query, batch_size)
      key = self.separate_heads(key, batch_size)
      value = self.separate_heads(value, batch_size)

      attention, weights = self.attention(query, key, value)

      attention = tf.transpose(attention, perm=[0, 2, 1, 3])
      concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

      output = self.combined_heads(concat_attention)
      return output

class TransformerBlock(layers.Layer):
  def __init__(self,embed_dim,num_heads,ff_dim,rate=0.1,**kwargs):
    super().__init__(**kwargs)
    self.att = MultiHeadSelfAttention(embed_dim,num_heads)
    self.ffn = tf.keras.Sequential(
        [
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ]
    )
    self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = layers.Dropout(rate)
    self.dropout2 = layers.Dropout(rate)
  def call(self,inputs,training):
    attn_output = self.att(inputs)
    attn_output = self.dropout1(attn_output,training=training)
    out1 = self.layernorm1(inputs + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output,training=training)
    return self.layernorm2(out1+ffn_output)

class CoAtNetStage(layers.Layer):
  def __init__(self,num_blocks,filters_in,filters_out,block_type,kernel_size=3,stride=1,expand_ratio=6,se_ratio=0.25,embed_dim=None,num_heads=None,ff_dim=None,activation='swish',**kwargs):
    super().__init__(**kwargs)
    self.block_type=block_type
    self.blocks=[]
    self.num_blocks = num_blocks

    if block_type=='conv':
      # The first MBConv block in a stage handles the stride (downsampling) and
      # potential channel change (filters_in to filters_out).
      self.blocks.append(MBConv(filters_in,filters_out,kernel_size=kernel_size,stride=stride,expand_ratio=expand_ratio,se_ratio=se_ratio,activation=activation, name=f'mbconv_block_0'))
      # Subsequent MBConv blocks in the same stage usually maintain resolution and channels (stride=1, filters_out=filters_out)
      for j in range(1, num_blocks):
        self.blocks.append(MBConv(filters_out,filters_out,kernel_size=kernel_size,stride=1,expand_ratio=expand_ratio,se_ratio=se_ratio,activation=activation, name=f'mbconv_block_{j}'))
    elif block_type=='transformer':
      if embed_dim is None or num_heads is None or ff_dim is None:
        raise ValueError("For transformer blocks, embed_dim, num_heads, and ff_dim must be provided.")
      for j in range(num_blocks):
        self.blocks.append(TransformerBlock(embed_dim,num_heads,ff_dim, name=f'transformer_block_{j}'))
    else:
      raise ValueError("block_type must be 'conv' or 'transformer'")

  def call(self,inputs,training=False): # Ensure training arg is here
    x=inputs
    for block in self.blocks:
        x=block(x, training=training) # Pass training arg to all blocks
    return x