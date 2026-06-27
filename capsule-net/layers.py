import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# 2. Define a custom 'squash' activation function
def squash(vectors):
    """
    The non-linear activation used in Capsule Networks.
    It squashes the length of the vector to be between 0 and 1.
    """
    # Calculate the squared Euclidean norm of the vectors along the last axis
    squared_norm = K.sum(K.square(vectors), axis=-1, keepdims=True)
    # Calculate the Euclidean norm of the vectors
    norm = K.sqrt(squared_norm + K.epsilon()) # Add K.epsilon to prevent division by zero

    # Apply the squash formula: v_j = (||s_j||^2 / (1 + ||s_j||^2)) * (s_j / ||s_j||)
    squash_factor = squared_norm / (1 + squared_norm) / norm

    # Multiply the vectors by the squash factor
    return squash_factor * vectors

class DigitCaps(Layer):
    def __init__(self, num_capsule, dim_capsule, routing_iterations=3, **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routing_iterations = routing_iterations

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(shape=(self.input_num_capsule,
                                        self.num_capsule,
                                        self.input_dim_capsule,
                                        self.dim_capsule),
                                name='W_digit_caps',
                                initializer='glorot_uniform')
        super(DigitCaps, self).build(input_shape)

    def call(self, inputs):
        u_hat = tf.einsum('bic,ijcd->bijd', inputs, self.W)
        b = tf.zeros(shape=(K.shape(inputs)[0], self.input_num_capsule, self.num_capsule))

        for i in range(self.routing_iterations):
            c = K.softmax(b, axis=-1)
            c_expanded = K.expand_dims(c, axis=-1)
            s = K.sum(c_expanded * u_hat, axis=1)
            v = squash(s)
            if i < self.routing_iterations - 1:
                v_expanded = K.expand_dims(v, axis=1)
                b += K.sum(u_hat * v_expanded, axis=-1)
        return v

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


# Masking layer logic
def Masking_layer(inputs,num_classes=10,dim_digit_capsule=16):
    digit_capsules_output = inputs[0]  # Output from DigitCaps layer (batch_size, num_classes, dim_digit_capsule)
    y_true_input = inputs[1]          # True labels (batch_size, num_classes)

    # Calculate the length of each digit capsule vector
    # ||v_j|| (batch_size, num_classes)
    v_length = K.sqrt(K.sum(K.square(digit_capsules_output), axis=-1, keepdims=True) + K.epsilon())

    # During training, y_true_input is the one-hot encoded true label.
    # During inference, y_true_input will be all zeros, and we select the capsule
    # with the largest length.

    # If y_true_input is all zeros (inference mode), use the capsule with the largest length
    # Otherwise (training mode), use y_true_input directly.

    # Create a mask for the 'winning' capsule during inference
    # K.argmax gives the index of the max length, K.one_hot converts to one-hot vector
    # The shape is (batch_size, num_classes, 1)
    inference_mask = K.one_hot(indices=K.argmax(v_length, 1), num_classes=num_classes)
    inference_mask = K.expand_dims(inference_mask, axis=-1)

    # Use tf.cond to choose between y_true_input and inference_mask based on whether y_true_input is all zeros
    # Check if y_true_input contains any non-zero values
    is_training = tf.cast(tf.reduce_sum(y_true_input), dtype=tf.bool)

    # Expand y_true_input to match capsule dimensions for element-wise multiplication
    y_true_input_expanded = K.expand_dims(y_true_input, axis=-1)

    # The mask to apply to the digit capsules output
    mask = tf.cond(is_training,
                   lambda: y_true_input_expanded,
                   lambda: inference_mask)

    # Apply the mask: only the chosen capsule's vector remains, others become zero
    masked_by_y = digit_capsules_output * mask

    # Flatten the masked output for the decoder
    # (batch_size, num_classes * dim_digit_capsule)
    return K.reshape(masked_by_y, [-1, num_classes * dim_digit_capsule])
