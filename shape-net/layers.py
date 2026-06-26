import tensorflow as tf

class TFMLPBlock(tf.keras.layers.Layer):
    def __init__(self, out_features, activation=None, **kwargs):
        super(TFMLPBlock, self).__init__(**kwargs)
        self.out_features = out_features
        self.activation = activation

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(units=self.out_features)
        if self.activation:
            self.activation_layer = tf.keras.layers.Activation(self.activation)
        else:
            self.activation_layer = tf.keras.layers.Identity() # No-op activation
        super(TFMLPBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.dense(inputs)
        return self.activation_layer(x)

    def get_config(self):
        config = super(TFMLPBlock, self).get_config()
        config.update({
            'out_features': self.out_features,
            'activation': tf.keras.activations.serialize(self.activation) # Serialize activation
        })
        return config

class TFSelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, num_heads=4, **kwargs):
        super(TFSelfAttentionLayer, self).__init__(**kwargs)
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

    def build(self, input_shape):
        # Query, Key, Value projections
        self.query_proj = TFMLPBlock(self.feature_dim, activation=None)
        self.key_proj = TFMLPBlock(self.feature_dim, activation=None)
        self.value_proj = TFMLPBlock(self.feature_dim, activation=None)

        # Output projection
        self.output_proj = TFMLPBlock(self.feature_dim, activation=None)
        super(TFSelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, num_points, feature_dim)
        batch_size = tf.shape(inputs)[0]
        num_points = tf.shape(inputs)[1]

        # Project inputs to Q, K, V
        q = self.query_proj(inputs)  # (batch_size, num_points, feature_dim)
        k = self.key_proj(inputs)    # (batch_size, num_points, feature_dim)
        v = self.value_proj(inputs)  # (batch_size, num_points, feature_dim)

        # Reshape for multi-head attention
        # (batch_size, num_points, num_heads, head_dim)
        q = tf.reshape(q, (batch_size, num_points, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, num_points, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, num_points, self.num_heads, self.head_dim))

        # Transpose to (batch_size, num_heads, num_points, head_dim) for batch matmul
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # Calculate attention scores: (batch_size, num_heads, num_points, num_points)
        # Using tf.einsum for efficient batch matrix multiplication
        attention_scores = tf.einsum('bhqd,bhkd->bhqk', q, k) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # Apply attention weights to values: (batch_size, num_heads, num_points, head_dim)
        attended_values = tf.einsum('bhqk,bhkd->bhqd', attention_weights, v)

        # Concatenate heads and reshape back to original feature_dim
        # (batch_size, num_points, feature_dim)
        attended_values = tf.transpose(attended_values, perm=[0, 2, 1, 3])
        attended_values = tf.reshape(attended_values, (batch_size, num_points, self.feature_dim))

        # Final output projection
        output = self.output_proj(attended_values)
        return output

    def get_config(self):
        config = super(TFSelfAttentionLayer, self).get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'num_heads': self.num_heads
        })
        return config