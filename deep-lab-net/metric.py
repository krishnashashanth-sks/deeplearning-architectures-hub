import tensorflow as tf

class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name='mean_io_u_from_logits', dtype=None):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)
        self._num_classes_custom = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true_squeezed = tf.squeeze(y_true, axis=-1)

        # Double-check clipping for safety
        y_true_squeezed = tf.clip_by_value(y_true_squeezed, 0, self._num_classes_custom - 1)
        y_pred_classes = tf.clip_by_value(y_pred_classes, 0, self._num_classes_custom - 1)

        return super().update_state(y_true_squeezed, y_pred_classes, sample_weight)
