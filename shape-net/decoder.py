import tensorflow as tf
from layers import *

class TFFoldingDecoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, output_num_points, grid_res=32, **kwargs):
        super(TFFoldingDecoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.output_num_points = output_num_points
        self.grid_res = grid_res

        # Generate base 2D grid using TensorFlow operations
        # Create a 1D tensor from 0 to 1 with `grid_res` points
        linear_space = tf.linspace(0.0, 1.0, grid_res)
        # Create a 2D grid from these 1D tensors
        grid_x, grid_y = tf.meshgrid(linear_space, linear_space)
        # Flatten and concatenate to form a (grid_res*grid_res, 2) tensor
        self.base_grid = tf.stack([tf.reshape(grid_x, [-1]), tf.reshape(grid_y, [-1])], axis=-1)

        if self.base_grid.shape[0] != output_num_points:
            raise ValueError(
                f"Grid resolution ({self.base_grid.shape[0]}) must match "
                f"output_num_points ({output_num_points}) for simplicity."
            )

        self.folding_mlps = [
            TFMLPBlock(256, activation='relu'),
            TFMLPBlock(256, activation='relu'),
            TFMLPBlock(3, activation=None) # Output 3D coordinates
        ]

    def build(self, input_shape): # input_shape here is the latent_vector shape
        # The first MLP block receives latent_dim + 2 (for grid coords) features
        self.folding_mlps[0].build(tf.TensorShape([None, self.latent_dim + 2]))
        # Build subsequent MLP blocks based on previous output shape
        self.folding_mlps[1].build(tf.TensorShape([None, 256]))
        self.folding_mlps[2].build(tf.TensorShape([None, 256]))
        super(TFFoldingDecoder, self).build(input_shape)

    def call(self, latent_vector):
        # latent_vector shape: (batch_size, latent_dim)
        batch_size = tf.shape(latent_vector)[0]

        # Repeat latent vector for each point in the grid
        # latent_repeated shape: (batch_size, output_num_points, latent_dim)
        latent_repeated = tf.expand_dims(latent_vector, 1) # (batch_size, 1, latent_dim)
        latent_repeated = tf.tile(latent_repeated, [1, self.output_num_points, 1])

        # Repeat base grid for each item in the batch
        # grid_repeated shape: (batch_size, output_num_points, 2)
        grid_repeated = tf.expand_dims(self.base_grid, 0) # (1, output_num_points, 2)
        grid_repeated = tf.tile(grid_repeated, [batch_size, 1, 1])

        # Concatenate latent vector and grid coordinates
        # inputs shape: (batch_size, output_num_points, latent_dim + 2)
        inputs = tf.concat([latent_repeated, grid_repeated], axis=-1)

        folded_points = inputs
        for mlp in self.folding_mlps:
            folded_points = mlp(folded_points)

        return folded_points # (batch_size, output_num_points, 3)

    def get_config(self):
        config = super(TFFoldingDecoder, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'output_num_points': self.output_num_points,
            'grid_res': self.grid_res
        })
        return config