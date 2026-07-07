from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

# ---  Custom Keras Layer for Type-1 Fuzzification (Example) ---
class FuzzificationLayer(layers.Layer):
    def __init__(self, num_linguistic_terms, input_dim, **kwargs):
        super(FuzzificationLayer, self).__init__(**kwargs)
        self.num_linguistic_terms = num_linguistic_terms
        self.input_dim = input_dim

    def build(self, input_shape):
        # Centers of the Gaussian MFs
        self.centers = self.add_weight(
            name='mf_centers',
            shape=(self.input_dim, self.num_linguistic_terms),
            initializer=keras.initializers.RandomUniform(minval=0., maxval=1.), # Initialize within a reasonable range
            trainable=True
        )
        # Log standard deviations to ensure positivity and for easier optimization
        self.log_std_devs = self.add_weight(
            name='mf_log_std_devs',
            shape=(self.input_dim, self.num_linguistic_terms),
            initializer=keras.initializers.RandomUniform(minval=-2., maxval=0.), # e.g., std_devs between exp(-2)=0.135 and exp(0)=1
            trainable=True
        )
        super(FuzzificationLayer, self).build(input_shape)

    def call(self, inputs):
        # Ensure std_devs are positive and avoid division by zero
        std_devs = tf.exp(self.log_std_devs) + keras.backend.epsilon()

        # Expand dimensions for broadcasting: (batch_size, input_dim, 1) vs (input_dim, num_linguistic_terms)
        expanded_inputs = tf.expand_dims(inputs, axis=-1)

        # Gaussian Membership Function calculation
        memberships = tf.exp(-0.5 * tf.square((expanded_inputs - self.centers) / std_devs))
        return memberships

    def get_config(self):
        config = super(FuzzificationLayer, self).get_config()
        config.update({
            'num_linguistic_terms': self.num_linguistic_terms,
            'input_dim': self.input_dim,
        })
        return config