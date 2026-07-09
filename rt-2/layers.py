import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_vision_encoder(input_shape,embedding_dim):
  inputs=keras.Input(shape=input_shape,name='image_input')
  x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(inputs)
  x=layers.MaxPooling2D((2,2))(x)
  x=layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
  x=layers.MaxPooling2D((2,2))(x)
  x=layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
  x=layers.MaxPooling2D((2,2))(x)
  x=layers.Flatten()(x)
  x=layers.Dense(512,activation='relu')(x)
  outputs=layers.Dense(embedding_dim,activation='relu',name='vision_embedding')(x)
  return keras.Model(inputs,outputs,name='vision_encoder')
# Implement MultiHeadSelfAttention as a custom layer
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.proj_dim = embed_dim // num_heads

    def build(self, input_shape):
        # Define Dense layers in build method to allow Keras to infer shapes
        self.query_dense = layers.Dense(self.embed_dim)
        self.key_dense = layers.Dense(self.embed_dim)
        self.value_dense = layers.Dense(self.embed_dim)
        self.combine_heads = layers.Dense(self.embed_dim)
        super(MultiHeadSelfAttention, self).build(input_shape)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def _split_heads(self, x): # Renamed and modified
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.proj_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x): # New helper method
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[2]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, seq_len, self.embed_dim))

    def call(self, inputs): # Modified to use new helper methods
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        attention, weights = self.attention(query, key, value)
        concat_attention = self._combine_heads(attention)
        output = self.combine_heads(concat_attention)
        return output

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape for this attention mechanism
        return input_shape

# Implement a Transformer Block as a custom layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim # Store embed_dim for build method
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.ff_dim, activation="relu"),
                layers.Dense(self.embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape for this transformer block
        return input_shape

# Implement the Language Encoder
def create_language_encoder(
    vocab_size,
    sequence_length,
    embed_dim,
    num_heads,
    ff_dim,
    num_transformer_blocks,
):
    inputs = keras.Input(shape=(sequence_length,), name='language_input')
    x = layers.Embedding(vocab_size, embed_dim)(inputs)

    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    # Pooling layer to get a fixed-size embedding
    # Using GlobalAveragePooling1D to get a single vector per sequence
    outputs = layers.GlobalAveragePooling1D(name='language_embedding')(x)

    return keras.Model(inputs, outputs, name='language_encoder')


def create_policy_transformer(fused_embedding_dim,
    action_dim,
    num_heads,
    ff_dim,
    num_transformer_blocks):
  inputs=keras.Input(shape=(fused_embedding_dim,),name='fused_vlm_features')
  x=layers.Reshape((1,fused_embedding_dim))(inputs)
  for _ in range(num_transformer_blocks):
    x=TransformerBlock(fused_embedding_dim,num_heads,ff_dim)(x)
  x=layers.Flatten()(x)
  outputs=layers.Dense(action_dim,activation='linear',name='robot_actions')(x)
  return keras.Model(inputs,outputs,name='policy_transformer')

