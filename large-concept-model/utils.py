import tensorflow as tf
import numpy as np

# ---  Optimizer and Learning Rate Schedule ---
class LinearWarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_learning_rate, warmup_steps, total_training_steps, **kwargs):
        super(LinearWarmupCosineDecay, self).__init__(**kwargs)
        self._peak_learning_rate = float(peak_learning_rate)
        self._warmup_steps = float(warmup_steps)
        self._total_training_steps = float(total_training_steps)
        self._decay_steps = self._total_training_steps - self._warmup_steps

    @tf.function
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        peak_learning_rate_tf = tf.constant(self._peak_learning_rate, dtype=tf.float32)
        warmup_steps_tf = tf.constant(self._warmup_steps, dtype=tf.float32)
        decay_steps_tf = tf.constant(self._decay_steps, dtype=tf.float32)

        return tf.cond(step < warmup_steps_tf,
                       lambda: peak_learning_rate_tf * (step / warmup_steps_tf),
                       lambda: peak_learning_rate_tf * 0.5 * (1 + tf.math.cos(np.pi * (step - warmup_steps_tf) / decay_steps_tf)))

    def get_config(self):
        return {
            "peak_learning_rate": self._peak_learning_rate,
            "warmup_steps": self._warmup_steps,
            "total_training_steps": self._total_training_steps,
        }

# ---  Transformer Encoder Block Components ---
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights
