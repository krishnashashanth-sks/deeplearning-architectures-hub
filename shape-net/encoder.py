import tensorflow as tf
from layers import *

class TFAdvanced3DEncoder(tf.keras.Model):
    def __init__(self, input_feature_dim, latent_dim=256, num_points=1024, **kwargs):
        super(TFAdvanced3DEncoder, self).__init__(**kwargs)
        self.input_feature_dim = input_feature_dim
        self.latent_dim = latent_dim
        self.num_points = num_points

        # First set of MLP blocks and Self-Attention
        self.sa_layer1_mlp_block1 = TFMLPBlock(64, activation='relu')
        self.sa_layer1_mlp_block2 = TFMLPBlock(128, activation='relu')
        self.sa_layer1_attention = TFSelfAttentionLayer(128) # Feature dim for SA

        # Second set of MLP blocks and Self-Attention
        self.sa_layer2_mlp_block1 = TFMLPBlock(256, activation='relu')
        self.sa_layer2_mlp_block2 = TFMLPBlock(512, activation='relu')
        self.sa_layer2_attention = TFSelfAttentionLayer(512) # Feature dim for SA

        # Global aggregator MLP blocks
        self.global_aggregator_mlp_block1 = TFMLPBlock(1024, activation='relu')
        self.global_aggregator_mlp_block2 = TFMLPBlock(latent_dim, activation=None)

    def call(self, point_cloud_features):
        # point_cloud_features shape: (batch_size, num_points, input_feature_dim)

        # Hierarchical Feature Learning (simplified pooling by slicing)
        # Layer 1: Process a subset of points (e.g., 1/4 of total points)
        features_l1 = point_cloud_features[:, ::4, :]
        features_l1 = self.sa_layer1_mlp_block1(features_l1)
        features_l1 = self.sa_layer1_mlp_block2(features_l1)
        features_l1 = self.sa_layer1_attention(features_l1)

        # Layer 2: Process a further subset of points (e.g., 1/4 of L1 points, 1/16 of total)
        features_l2 = features_l1[:, ::4, :]
        features_l2 = self.sa_layer2_mlp_block1(features_l2)
        features_l2 = self.sa_layer2_mlp_block2(features_l2)
        features_l2 = self.sa_layer2_attention(features_l2)

        # Global aggregation: Max pooling across points
        global_feature = tf.reduce_max(features_l2, axis=1) # (batch_size, 512)

        # Final MLP to get the latent vector
        latent_vector = self.global_aggregator_mlp_block1(global_feature)
        latent_vector = self.global_aggregator_mlp_block2(latent_vector)

        return latent_vector

    def get_config(self):
        config = super(TFAdvanced3DEncoder, self).get_config()
        config.update({
            'input_feature_dim': self.input_feature_dim,
            'latent_dim': self.latent_dim,
            'num_points': self.num_points
        })
        return config