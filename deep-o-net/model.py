import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the DeepONet class by combining Branch and Trunk networks
class DeepONet(keras.Model):
    def __init__(self, branch_net, trunk_net):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        # The bias term is a learnable scalar added to the final output
        self.bias = tf.Variable(tf.zeros(1), trainable=True)

    def call(self, inputs):
        branch_input, trunk_input = inputs
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        # Element-wise multiplication followed by summation, then add bias
        # This implements: G(u)(x) = sum(B_i(u) * T_i(x)) + bias
        output = tf.reduce_sum(branch_output * trunk_output, axis=1, keepdims=True) + self.bias
        return output
