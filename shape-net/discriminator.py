import tensorflow as tf

class TFDiscriminator(tf.keras.Model):
    def __init__(self, input_shape, **kwargs):
        super(TFDiscriminator, self).__init__(**kwargs)
        self.input_shape_ = input_shape  # Store original input shape for flattening
        flattened_input_dim = input_shape[0] * input_shape[1]

        self.mlp_layers = [
            TFMLPBlock(flattened_input_dim, activation='relu'),
            TFMLPBlock(256, activation='relu'),
            TFMLPBlock(128, activation='relu'),
            TFMLPBlock(1, activation='sigmoid') # Output a single probability
        ]

    def call(self, inputs):
        # inputs shape: (batch_size, num_points, num_features)
        # Flatten the input point cloud
        flattened_inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1])

        x = flattened_inputs
        for i, mlp in enumerate(self.mlp_layers):
            # The first MLPBlock receives the flattened_input_dim
            # Subsequent MLPBlocks receive the output_features of the previous one
            if i == 0:
                # Pass the flattened_input_dim to the first MLPBlock's call method
                x = mlp(x)
            else:
                x = mlp(x)
        return x

    def get_config(self):
        config = super(TFDiscriminator, self).get_config()
        config.update({
            'input_shape': self.input_shape_
        })
        return config