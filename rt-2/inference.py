import tensorflow as tf
import numpy as np

def rt2_inference(model, image, language_instruction,vocab_size,sequence_length):
    processed_image = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), axis=0)

    dummy_tokenized_language = np.random.randint(0, vocab_size, (1, sequence_length), dtype=np.int32)
    processed_language = tf.convert_to_tensor(dummy_tokenized_language, dtype=tf.int32)

    # Make prediction
    predicted_actions = model([processed_image, processed_language], training=False)

    # Convert predictions to a numpy array and return
    return predicted_actions.numpy().squeeze()